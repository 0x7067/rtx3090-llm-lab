"""Native full-page GLM plus pinned SDK layout/cropping and paired table tasks.

The serial scheduler is explicit. SDK formatting/Markdown and hosted MaaS are
not used; raw model strings and real source-region bounds remain evidence.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from contextlib import closing
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import time

from ..paddle.errors import WorkerError
from ..paddle.decode import decode_pages
from .config import GlmWorkerConfig
from .identifiers import plan_rereads
from .regions import plan_regions
from .cpu_reader import CPU_SECONDS, READER_ID, read_cpu, reader_identity, validate_observation

MAX_CALLS = 256
MAX_TEXT = 1_000_000
MAX_BLOCK_TEXT = 10_000
MAX_BLOCKS = 50_000
MAX_PAGE_BLOCKS = 10_000
MAX_TOKENS = 4096
REQUEST_SECONDS = 60
MAX_REQUEST_BYTES = 64 * 1024 * 1024
# The fixed native request metadata is below 4096 bytes. Larger images retain
# the previous bounded SDK preparation path instead of raising transport limits.
MAX_NATIVE_IMAGE_URI = MAX_REQUEST_BYTES - 4096
TEXT_CONTEXT_PIXELS = 6
IMAGE_LABELS = {"image", "header_image", "footer_image"}
MAX_PENDING_RECOVERIES = 100  # At most two metadata records on each of 50 pages.
MAX_CPU_READS = 2
RECOVERY_REVIEW_NOTE = (
    "An identifier has differing or unexpected readings. Additional source readings, "
    "when available, are retained for human review; no value is corrected or selected."
)
PROMPTS = {"text":"Text Recognition:","table":"Table Recognition:","formula":"Formula Recognition:"}
EXPECTED_VERSIONS = {"glmocr":"0.1.5","torch":"2.10.0+cpu","torchvision":"0.25.0+cpu",
                     "transformers":"5.4.0","Pillow":"13.0.0","numpy":"2.2.6","pypdfium2":"4.30.0"}


def native_image_request(image,task):
    if task not in PROMPTS:raise WorkerError("invalid_provider_output")
    with io.BytesIO() as encoded:
        image.save(encoded,format="PNG")
        uri="data:image/png;base64,"+base64.b64encode(encoded.getvalue()).decode("ascii")
    return {"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":uri}},
        {"type":"text","text":PROMPTS[task]}]}],"max_tokens":MAX_TOKENS,"temperature":0,"top_p":1}


def region_request(image,task,loader):
    # Native GLM already preprocesses its images. Reapplying the SDK's patch/min-
    # pixel resize to a small crop can stretch its glyphs (172x40 -> 168x56).
    # Preserve the exact crop pixels, as the established whole-page path does.
    request=native_image_request(image,task)
    # Keep the previous SDK crop request's fixed sampling parameters too.
    request.update(top_k=1,repetition_penalty=1.0)
    if len(request["messages"][0]["content"][0]["image_url"]["url"])>MAX_NATIVE_IMAGE_URI:
        return loader.build_request_from_image(image,task)
    return request


def source_crop_pixels(region,width,height):
    # Match the pinned SDK's int(normalized * pixels / 1000) crop exactly.
    # Detector-normalized floats alone can exclude edge pixels on small images.
    left,top,right,bottom=region["bbox_2d"]
    x1,x2=int(left*width/1000),int(right*width/1000)
    y1,y2=int(top*height/1000),int(bottom*height/1000)
    if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise WorkerError("invalid_provider_output")
    return x1,y1,x2,y2


def normalized_box(pixels,width,height):
    x1,y1,x2,y2=pixels
    x,y=x1/width,y1/height
    return {"x":x,"y":y,"width":min((x2-x1)/width,1-x),"height":min((y2-y1)/height,1-y)}


def source_crop_box(region,width,height):
    return normalized_box(source_crop_pixels(region,width,height),width,height)


def prepare_region(image,region,descriptor,sdk_crop):
    pixels=source_crop_pixels(region,image.width,image.height)
    # Paragraph predictions can end inside a glyph. Retain a small rectangle of
    # real surrounding pixels rather than stretching or white-padding the crop.
    # Table/formula/image reads and paired table counter-reads remain unchanged.
    if (descriptor["origin"]=="native" and region["task_type"]=="text"
            and region["label"] not in IMAGE_LABELS):
        x1,y1,x2,y2=pixels
        pixels=(max(0,x1-TEXT_CONTEXT_PIXELS),max(0,y1-TEXT_CONTEXT_PIXELS),
                min(image.width,x2+TEXT_CONTEXT_PIXELS),min(image.height,y2+TEXT_CONTEXT_PIXELS))
        return image.crop(pixels),normalized_box(pixels,image.width,image.height)
    return sdk_crop(image,region["bbox_2d"],region.get("polygon")),normalized_box(pixels,image.width,image.height)


def chunks(text: str, table: bool) -> list[str]:
    if not isinstance(text,str):raise WorkerError("invalid_provider_output")
    try:text.encode("utf-8","strict")
    except UnicodeError:raise WorkerError("invalid_provider_output") from None
    if len(text)>MAX_TEXT:raise WorkerError("resource_limit",resource_limit="total_text")
    if not text.strip():return []
    if table:
        if len(text)>MAX_BLOCK_TEXT:raise WorkerError("resource_limit",resource_limit="block_text")
        return [text]
    result=[];pending=""
    for line in text.splitlines(keepends=True):
        if len(line)>MAX_BLOCK_TEXT:raise WorkerError("resource_limit",resource_limit="block_text")
        if len(pending)+len(line)>MAX_BLOCK_TEXT:
            result.append(pending);pending=""
        pending+=line
    if pending:result.append(pending)
    return result


class NativeClient:
    def __init__(self,config: GlmWorkerConfig,deadline: float):
        import requests
        from requests.adapters import HTTPAdapter
        self.requests=requests;self.deadline=deadline;self.calls=0;self.config=config
        self.session=requests.Session();self.session.trust_env=False
        self.session.mount("http://",HTTPAdapter(max_retries=0,pool_connections=1,pool_maxsize=1))

    def recognize(self,request: dict) -> str:
        if self.calls>=MAX_CALLS:raise WorkerError("resource_limit",resource_limit="region_count")
        if time.monotonic()+REQUEST_SECONDS+10>self.deadline:raise WorkerError("timeout",True)
        request={**request,"model":self.config.served_model_name,"stream":False,"seed":0}
        # The pinned SDK uses the Hugging Face/vLLM spelling. llama.cpp accepts
        # repeat_penalty and silently ignores repetition_penalty. Translate at
        # the transport boundary, including requests from the SDK size fallback.
        if "repetition_penalty" in request:
            if "repeat_penalty" in request:raise WorkerError("invalid_provider_output")
            request["repeat_penalty"]=request.pop("repetition_penalty")
        request.setdefault("repeat_penalty",1.0)
        penalty=request["repeat_penalty"]
        if type(penalty) not in {int,float} or not 1.0<=penalty<=2.0:
            raise WorkerError("invalid_provider_output")
        if (request.get("max_tokens")!=MAX_TOKENS or request.get("temperature")!=0
                or request.get("top_p")!=1):raise WorkerError("invalid_provider_output")
        messages=request.get("messages")
        if not isinstance(messages,list) or len(messages)!=1 or not isinstance(messages[0],dict) or messages[0].get("role")!="user":
            raise WorkerError("invalid_provider_output")
        content=messages[0].get("content")
        if not isinstance(content,list) or len(content)!=2 or not all(isinstance(p,dict) for p in content):
            raise WorkerError("invalid_provider_output")
        images=[p for p in content if p.get("type")=="image_url"]
        texts=[p for p in content if p.get("type")=="text"]
        if (len(images)!=1 or len(texts)!=1 or texts[0].get("text") not in PROMPTS.values()
                or not isinstance(images[0].get("image_url"),dict)
                or not isinstance(images[0]["image_url"].get("url"),str)
                or not images[0]["image_url"]["url"].startswith("data:image/png;base64,")):
            raise WorkerError("invalid_provider_output")
        encoded=json.dumps(request,separators=(",",":"),ensure_ascii=False).encode()
        if len(encoded)>MAX_REQUEST_BYTES:raise WorkerError("resource_limit",resource_limit="input_bytes")
        self.calls+=1
        try:
            with self.session.post(self.config.native_base_url+"/chat/completions",data=encoded,
                    headers={"Content-Type":"application/json"},allow_redirects=False,stream=True,
                    timeout=(3,REQUEST_SECONDS)) as response:
                if response.status_code!=200:
                    raise WorkerError("provider_unavailable" if response.status_code>=500 or response.status_code==429 else "provider_failure",
                                      response.status_code>=500 or response.status_code==429)
                body=bytearray()
                for part in response.iter_content(65536):
                    if time.monotonic()>self.deadline:raise WorkerError("timeout",True)
                    if len(body)+len(part)>1_048_576:raise WorkerError("invalid_provider_output")
                    body.extend(part)
                result=json.loads(body.decode("utf-8","strict"))
        except WorkerError:raise
        except self.requests.Timeout:raise WorkerError("timeout",True) from None
        except self.requests.RequestException:raise WorkerError("provider_unavailable",True) from None
        except (ValueError,UnicodeError):raise WorkerError("invalid_provider_output") from None
        if not isinstance(result,dict):raise WorkerError("invalid_provider_output")
        choices=result.get("choices")
        if not isinstance(choices,list) or len(choices)!=1:raise WorkerError("invalid_provider_output")
        choice=choices[0]
        if not isinstance(choice,dict):raise WorkerError("invalid_provider_output")
        if choice.get("finish_reason")=="length":raise WorkerError("resource_limit",resource_limit="token_limit")
        message=choice.get("message")
        if (choice.get("finish_reason")!="stop" or not isinstance(message,dict)
                or message.get("tool_calls") or message.get("function_call") or message.get("refusal")):
            raise WorkerError("invalid_provider_output")
        text=message.get("content")
        if not isinstance(text,str):raise WorkerError("invalid_provider_output")
        try:text.encode("utf-8","strict")
        except UnicodeError:raise WorkerError("invalid_provider_output") from None
        return text

    def close(self):self.session.close()


def make_sdk(config: GlmWorkerConfig):
    try:
        if any(importlib.metadata.version(k)!=v for k,v in EXPECTED_VERSIONS.items()):
            raise WorkerError("provider_unavailable")
    except importlib.metadata.PackageNotFoundError:
        raise WorkerError("provider_unavailable") from None
    import torch
    import yaml
    import glmocr
    from glmocr.config import GlmOcrConfig
    from glmocr.dataloader import PageLoader
    from glmocr.layout import PPDocLayoutDetector
    from glmocr.utils.image_utils import crop_image_region
    if torch.version.cuda is not None or torch.cuda.is_available():raise WorkerError("provider_unavailable")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cfg=yaml.safe_load((Path(glmocr.__file__).parent/"config.yaml").read_text())
    pipeline=cfg["pipeline"]
    pipeline["maas"].update(enabled=False,api_key=None,retry_max_attempts=0)
    pipeline["page_loader"].update(max_tokens=MAX_TOKENS,temperature=0.0,top_p=1.0,top_k=1,
        repetition_penalty=1.0,image_format="PNG",task_prompt_mapping=PROMPTS.copy())
    layout=pipeline["layout"]
    layout.update(model_dir=config.layout_model_dir,device="cpu",cuda_visible_devices="",workers=1,batch_size=1,use_polygon=True)
    labels=set(layout["id2label"].values())
    layout["label_task_mapping"]={"text":sorted(labels-{"table","display_formula","inline_formula"}),
        "table":["table"],"formula":["display_formula","inline_formula"],"skip":[],"abandon":[]}
    cfg["logging"]["level"]="ERROR"
    parsed=GlmOcrConfig.model_validate(cfg)
    if parsed.pipeline.maas.enabled:raise WorkerError("provider_unavailable")
    detector=PPDocLayoutDetector(parsed.pipeline.layout)
    detector.start()
    return detector,PageLoader(parsed.pipeline.page_loader),crop_image_region


class GlmAdapter:
    def __init__(self,config,budget,sdk_factory=make_sdk,client_factory=NativeClient,*,source_sha256=None,
                 cpu_factory=read_cpu,work_dir=None):
        self.budget=budget;self.text_used=0;self.block_count=0
        self.source_sha256=source_sha256
        self._recovery_pending=[];self._recovery_finished=False
        self._cpu_pending=[];self.cpu_reads=0;self.first_failure=None
        self.cpu_model_dir=getattr(config,"cpu_model_dir",None);self.cpu_factory=cpu_factory
        self.work_dir=os.path.abspath(work_dir or os.getcwd())
        if self.cpu_model_dir and (not isinstance(source_sha256,str) or len(source_sha256)!=64
                                  or any(c not in "0123456789abcdef" for c in source_sha256)):
            raise WorkerError("invalid_document")
        self.deadline=time.monotonic()+270
        self.client=client_factory(config,self.deadline)
        try:self.detector,self.loader,self.crop=sdk_factory(config)
        except BaseException:
            self.client.close()
            raise

    def close(self):
        self._recovery_pending.clear()
        self._cpu_pending.clear()
        try:self.client.close()
        finally:self.detector.stop()

    @staticmethod
    def _recovery_note(page):
        # Merge into the existing bounded human-review warning. A 50-page
        # document must not exceed the unchanged 100-warning protocol cap.
        for warning in page["warnings"]:
            if warning["code"]=="low_quality":
                if RECOVERY_REVIEW_NOTE not in warning["message"]:
                    combined=warning["message"]+" "+RECOVERY_REVIEW_NOTE
                    if len(combined)<=500:warning["message"]=combined
                return
        if len(page["warnings"])<2:
            page["warnings"].append({"code":"low_quality","message":RECOVERY_REVIEW_NOTE,"page":page["page"]})

    def _queue_recovery(self,page,whole_text,records):
        plan=plan_rereads(whole_text,[record["hint"] for record in records])
        if plan["requests"] or plan["limited"]:self._recovery_note(page)
        by_key={record["hint"]["key"]:record for record in records}
        for request in plan["requests"]:
            if len(self._recovery_pending)>=MAX_PENDING_RECOVERIES:break
            record=by_key[request["key"]]
            self._recovery_pending.append({"page":page,"dimensions":(page["width"],page["height"]),
                                           "region":deepcopy(record["region"])})

    def _append_recovery(self,page,text,box):
        # Check all output bounds before modifying even one original block.
        pieces=chunks(text,False)
        if not pieces:return
        count=len(text)+(1 if page["text"] else 0)
        if (self.text_used+count>MAX_TEXT or self.block_count+len(pieces)>MAX_BLOCKS
                or len(page["blocks"])+len(pieces)>MAX_PAGE_BLOCKS):
            raise WorkerError("resource_limit",resource_limit="total_text")
        for piece in pieces:
            page["blocks"].append({"id":f"b{len(page['blocks'])+1}","text":piece,"box":box.copy()})
        page["text"]+=("\n" if page["text"] else "")+text
        self.text_used+=count;self.block_count+=len(pieces)

    @staticmethod
    def _diagnostic(error):
        if isinstance(error,MemoryError):return {"code":"resource_limit","resourceLimit":"memory"}
        allowed={"resource_limit","timeout","provider_unavailable","provider_failure","invalid_provider_output"}
        code=error.code if isinstance(error,WorkerError) and error.code in allowed else "provider_failure"
        return {"code":code,"resourceLimit":error.resource_limit if code=="resource_limit" else None}

    def _cpu_record(self,item,outcome,failure,observation=None,block_ids=None):
        item["page"]["auxiliaryReading"]={"readerId":READER_ID,
            "recipeFingerprint":reader_identity()["fingerprint"],"sourceSha256":self.source_sha256,
            "trigger":deepcopy(item["trigger"]),"outcome":outcome,
            "raster":dict(observation["input"]) if observation is not None else None,
            "failure":failure,"blockIds":block_ids or []}

    def _append_cpu(self,item,reference,observation):
        # Independent boundary check binds the component response to this page,
        # even if a future implementation changes the subprocess transport.
        area=reference.width*reference.height
        with self.budget.memory(area):
            self.budget.reserve(area)
            rgb_sha=hashlib.sha256(reference.image.tobytes()).hexdigest()
        expected={"width":reference.width,"height":reference.height,"rgbSha256":rgb_sha,
                  "pngSha256":observation.get("input",{}).get("pngSha256")}
        validate_observation(observation,expected)
        lines=[line for line in observation["lines"] if line["text"].strip()]
        if not lines:return False
        page=item["page"];text=observation["text"]
        count=len(text)+(1 if page["text"] else 0)
        if (self.text_used+count>MAX_TEXT or self.block_count+len(lines)>MAX_BLOCKS
                or len(page["blocks"])+len(lines)>MAX_PAGE_BLOCKS):
            raise WorkerError("resource_limit",resource_limit="total_text")
        blocks=[]
        for i,line in enumerate(lines,1):
            xs=[point[0] for point in line["polygon"]];ys=[point[1] for point in line["polygon"]]
            blocks.append({"id":f"cpu-b{len(page['blocks'])+i}","text":line["text"],
                "box":normalized_box((min(xs),min(ys),max(xs),max(ys)),reference.width,reference.height),
                "readerId":READER_ID,"recognitionScore":line["score"]})
        # All checks precede mutation; failed optional output cannot damage primary evidence.
        combined_blocks=page["blocks"]+blocks
        combined_text=page["text"]+("\n" if page["text"] else "")+text
        record={**page["auxiliaryReading"],"outcome":"read","failure":None,
                "raster":dict(observation["input"]),"blockIds":[block["id"] for block in blocks]}
        new_text_used=self.text_used+count;new_block_count=self.block_count+len(blocks)
        page.update(blocks=combined_blocks,text=combined_text,status="read",reason=None,auxiliaryReading=record)
        self.text_used=new_text_used;self.block_count=new_block_count
        return True

    def _finish_cpu(self,pending,data,source,pdf_scale):
        for index,item in enumerate(pending):
            diagnostic={"code":"timeout","resourceLimit":None} if index<MAX_CPU_READS else {
                "code":"resource_limit","resourceLimit":"region_count"}
            self._cpu_record(item,"skipped",diagnostic)
        if not pending or time.monotonic()+CPU_SECONDS+5>self.deadline:return
        selected=pending[:MAX_CPU_READS]
        wanted=frozenset(item["page"]["page"] for item in selected)
        try:
            with closing(decode_pages(data,source["mime"],self.budget,pdf_scale,selected_pages=wanted)) as references:
                for reference in references:
                    try:
                        item=next((item for item in selected if item["page"]["page"]==reference.number),None)
                        if item is None:continue
                        if time.monotonic()+CPU_SECONDS+5>self.deadline:return
                        if reference.image is None or (reference.width,reference.height)!=item["dimensions"]:
                            self._cpu_record(item,"skipped",{"code":"invalid_provider_output","resourceLimit":None})
                            continue
                        if self.text_used>=MAX_TEXT or self.block_count>=MAX_BLOCKS or len(item["page"]["blocks"])>=MAX_PAGE_BLOCKS:
                            self._cpu_record(item,"skipped",{"code":"resource_limit","resourceLimit":"total_text"})
                            continue
                        try:
                            self.cpu_reads+=1
                            observation=self.cpu_factory(reference.image,self.cpu_model_dir,self.work_dir,self.budget,self.deadline)
                            if not self._append_cpu(item,reference,observation):
                                self._cpu_record(item,"failed",{"code":"no_text","resourceLimit":None})
                        except Exception as error:
                            self._cpu_record(item,"failed",self._diagnostic(error))
                    finally:reference.close()
        except Exception as error:
            for item in selected:
                if item["page"]["auxiliaryReading"]["outcome"]=="skipped":
                    self._cpu_record(item,"skipped",self._diagnostic(error))
            return

    def finish(self,pages,data,source,pdf_scale=2):
        if self._recovery_finished:return
        self._recovery_finished=True
        pending=self._recovery_pending;self._recovery_pending=[]
        cpu=self._cpu_pending;self._cpu_pending=[]
        if not pending and not cpu:return
        if any(not any(item["page"] is page for page in pages) for item in pending+cpu):return
        if any(item["dimensions"]!=(item["page"]["width"],item["page"]["height"]) for item in pending+cpu):return
        if (not self.source_sha256 or source["originalSha256"]!=self.source_sha256
                or len(data)!=source["sizeBytes"] or hashlib.sha256(data).hexdigest()!=self.source_sha256
                or self.budget.live):return
        if cpu:self._finish_cpu(cpu,data,source,pdf_scale)
        self._finish_identifiers(pending,pages,data,source,pdf_scale)

    def _finish_identifiers(self,pending,pages,data,source,pdf_scale=2):
        """Run optional reads only after every primary page has finished.

        Pending work is metadata, not retained rasters/encoded images. Selected
        pages are re-decoded from the same checked original under the remaining
        raster, deadline, call and output bounds. Failure leaves primary reads.
        """
        if not pending:return
        if any(not any(item["page"] is page for page in pages) for item in pending):return
        try:
            if (not self.source_sha256 or source["originalSha256"]!=self.source_sha256
                    or len(data)!=source["sizeBytes"] or hashlib.sha256(data).hexdigest()!=self.source_sha256
                    or self.budget.live or self.client.calls>=MAX_CALLS
                    or time.monotonic()+REQUEST_SECONDS+10>self.deadline):return
            wanted=frozenset(item["page"]["page"] for item in pending)
            with closing(decode_pages(data,source["mime"],self.budget,pdf_scale,selected_pages=wanted)) as references:
                for reference in references:
                    try:
                        for item in pending:
                            if item["page"]["page"]!=reference.number:continue
                            if reference.image is None or (reference.width,reference.height)!=item["dimensions"]:continue
                            if self.client.calls>=MAX_CALLS or time.monotonic()+REQUEST_SECONDS+10>self.deadline:return
                            try:
                                pixels=source_crop_pixels(item["region"],reference.width,reference.height)
                                area=(pixels[2]-pixels[0])*(pixels[3]-pixels[1])
                                # The pinned polygon crop holds three RGB surfaces
                                # and a grayscale mask; include encoder scratch too.
                                with self.budget.memory(4*area):
                                    self.budget.reserve(4*area)
                                    crop=self.crop(reference.image,item["region"]["bbox_2d"],item["region"].get("polygon"))
                                    try:request=region_request(crop,"text",self.loader)
                                    finally:crop.close()
                                    text=self.client.recognize(request)
                                self._append_recovery(item["page"],text,normalized_box(pixels,reference.width,reference.height))
                            except WorkerError as error:
                                if error.code in {"timeout","provider_unavailable"}:return
                            except Exception:
                                # This optional path cannot discard primary source evidence.
                                continue
                    finally:reference.close()
        except Exception:
            # Original read results and their human-review warning stay intact.
            return

    def recognize(self,page):
        if self._recovery_finished:raise WorkerError("provider_failure")
        if page.image is None:
            reason=page.reason or "The original page could not be rendered."
            return {"page":page.number,"width":page.width,"height":page.height,"status":"unreadable",
                    "reason":reason,"text":"","blocks":[],"warnings":[{"code":"page_unreadable","message":reason,"page":page.number}]}
        blocks=[];readings=[];warnings=[];first_error=None;whole_text="";records=[];trigger=None
        def failed(error,stage):
            nonlocal trigger
            if trigger is None:trigger={"stage":stage,"diagnostic":self._diagnostic(error)}
        def warn(message,unreadable=False):
            code="page_unreadable" if unreadable else "low_quality"
            if not any(w["code"]==code for w in warnings):warnings.append({"code":code,"message":message,"page":page.number})
        def retain(text,box,table=False):
            pieces=chunks(text,table)
            if not pieces:return False
            count=len(text)+(1 if readings else 0)
            if (self.text_used+count>MAX_TEXT or self.block_count+len(pieces)>MAX_BLOCKS
                    or len(blocks)+len(pieces)>MAX_PAGE_BLOCKS):
                raise WorkerError("resource_limit",resource_limit="total_text")
            prefix="table-" if table else ""
            for piece in pieces:blocks.append({"id":f"{prefix}b{len(blocks)+1}","text":piece,"box":box.copy()})
            readings.append(text);self.text_used+=count;self.block_count+=len(pieces)
            return True
        warn("Whole-page and layout-guided OCR readings may disagree or omit content. Locations are source-region predictions, not word/cell guarantees; human review remains required.")
        # The established native whole-page path has no external SDK resizing.
        try:
            request=native_image_request(page.image,"text")
            whole_text=self.client.recognize(request)
            retain(whole_text,{"x":0,"y":0,"width":1,"height":1})
            del request
        except WorkerError as error:
            first_error=error
            failed(error,"whole_page")
            warn("The whole-page read did not complete; only successful source-region readings can be retained.",True)
        try:
            native,_=self.detector.process([page.image],save_visualization=False,global_start_idx=0,use_polygon=True)
            if not isinstance(native,list) or len(native)!=1:raise WorkerError("invalid_provider_output")
            planned,descriptors,stats=plan_regions(native[0])
            if stats["invalid"] or stats["limited"]:
                if trigger is None:trigger={"stage":"layout","diagnostic":{"code":"layout_failure","resourceLimit":None}}
                warn("Layout output exceeded supported bounds or contained invalid regions; complete source coverage is not established.",True)
            for region,descriptor in zip(planned,descriptors):
                try:
                    if time.monotonic()+REQUEST_SECONDS+10>self.deadline:
                        raise WorkerError("timeout",True)
                    crop,box=prepare_region(page.image,region,descriptor,self.crop)
                    try:request=region_request(crop,region["task_type"],self.loader)
                    finally:crop.close()
                    text=self.client.recognize(request)
                    retain(text,box,region["task_type"]=="table")
                    records.append({"region":region,"hint":{"key":str(descriptor["index"]),"text":text,"box":box,
                        "origin":descriptor["origin"],"task":descriptor["task"],"label":region["label"]}})
                except WorkerError as error:
                    if first_error is None:first_error=error
                    failed(error,"layout_region")
                    warn("A layout-guided reading did not complete or fit the output bounds; successful original readings remain retained.",True)
                    if error.code=="timeout" or error.resource_limit in {"region_count","total_text"}:break
                except Exception:
                    failed(None,"layout_region")
                    warn("A source region could not be processed; successful original readings remain retained.",True)
        except Exception:
            if trigger is None:trigger={"stage":"layout","diagnostic":{"code":"layout_failure","resourceLimit":None}}
            warn("Layout analysis did not complete; successful whole-page text remains retained.",True)
        if not blocks:
            if first_error is not None:
                if not self.cpu_model_dir:raise first_error
                if self.first_failure is None:self.first_failure=first_error
            extrema=page.image.getextrema()
            if first_error is not None or not all(low==high and low>=250 for low,high in extrema):
                reason="No supported text could be recognized on the original page."
                if not self.cpu_model_dir:
                    return {"page":page.number,"width":page.width,"height":page.height,"status":"unreadable",
                            "reason":reason,"text":"","blocks":[],"warnings":[{"code":"page_unreadable","page":page.number,"message":reason}]}
                warn(reason,True)
                if trigger is None:trigger={"stage":"no_text","diagnostic":{"code":"no_text","resourceLimit":None}}
        result={"page":page.number,"width":page.width,"height":page.height,"status":"read","reason":None,
                "text":"\n".join(readings),"blocks":blocks,"warnings":warnings}
        if self.cpu_model_dir and trigger is not None:
            if not blocks:result.update(status="unreadable",reason="Primary OCR could not read supported text on the original page.",text="")
            self._cpu_pending.append({"page":result,"dimensions":(page.width,page.height),"trigger":trigger})
        self._queue_recovery(result,whole_text,records)
        return result
