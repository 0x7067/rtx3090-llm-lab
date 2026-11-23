"""Bounded format/disagreement hints for additive source-region OCR reads.

No expected answer, correction, checksum or preferred value is produced. The
caller must supply actual native task/origin metadata from the current page.
"""
from __future__ import annotations
from copy import deepcopy
import math,re

MAX_TEXT=10_000
MAX_REGIONS=128
MAX_EXTRA_READS=2
MAX_HASH_ANCHORS=4
MAX_OPAQUE_TOKENS=32
FULL={"x":0,"y":0,"width":1,"height":1}
POSTAL_LINE=re.compile(r"^[ \t]*OBJETO[ \t.:=>-]{1,24}([A-Za-z0-9][A-Za-z0-9 \t]{9,26}[A-Za-z0-9])[ \t]*$",re.ASCII|re.I)
POSTAL_SHAPE=re.compile(r"[A-Z]{2}[0-9]{9}[A-Z]{2}",re.ASCII)
HASH_LINE=re.compile(r"^[ \t]*MD[- ]?5[ \t]*:[ \t]*([A-Za-z0-9]{28,36})[ \t]*$",re.ASCII|re.I)
HEX32=re.compile(r"[0-9A-Fa-f]{32}",re.ASCII)
OPAQUE32=re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{32}(?![A-Za-z0-9])",re.ASCII)

def lines(text,maximum=MAX_TEXT):
    if not isinstance(text,str) or len(text)>maximum:return []
    result=[];offset=0
    for line in text.splitlines(keepends=True):
        value=line.rstrip('\r\n')
        if len(value)<=512:result.append((offset,value))
        offset+=len(line)
    return result

def observations(text,maximum=MAX_TEXT):
    """Keep the exact literal, quote and offsets; comparisons are separate."""
    out=[]
    for offset,line in lines(text,maximum):
        for kind,pattern in [('postal-shape',POSTAL_LINE),('md5-shape',HASH_LINE)]:
            m=pattern.fullmatch(line)
            if not m:continue
            raw=m[1];comparison=raw.replace(' ','').replace('\t','').upper()
            if kind=='postal-shape' and not 11<=len(comparison)<=15:continue
            shape=bool((POSTAL_SHAPE if kind=='postal-shape' else HEX32).fullmatch(comparison))
            out.append({'kind':kind,'literal':raw,'comparison':comparison,'shapeMatches':shape,
                        'start':offset+m.start(1),'end':offset+m.end(1),'quote':line})
    return out

def box_ok(box):
    if not isinstance(box,dict) or set(box)!={'x','y','width','height'}:return False
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in box.values()):return False
    return (0<=box['x']<1 and 0<=box['y']<1 and 0<box['width']<=1-box['x']+1e-12
            and 0<box['height']<=1-box['y']+1e-12)

def plan_rereads(whole_text,regions):
    """Return at most two source-region requests, never preferred/corrected text.

    It considers only native ordinary text, not tables,
    paired reads, images, formulas, unknown provenance, oversized text or pages.
    A same-page MD5 label can anchor a nearby 32-character disagreement even
    when the smaller reading's label is itself wrong. Hamming distance only
    chooses another image read; it never manufactures a value or repairs a label.
    """
    if not isinstance(whole_text,str) or len(whole_text)>1_000_000 or not isinstance(regions,list) or len(regions)>MAX_REGIONS:
        return {'requests':[],'limited':True,'reason':'invalid-or-oversize-inventory'}
    # Only whole-page physical lines supply anchors; no joined/cross-page roles.
    anchors=sorted({o['comparison'] for o in observations(whole_text,maximum=1_000_000)
                    if o['kind']=='md5-shape' and len(o['comparison'])==32})
    hash_limited=len(anchors)>MAX_HASH_ANCHORS
    if hash_limited:anchors=[]
    requests=[];limited=hash_limited;seen=set()
    for region in regions:
        if not isinstance(region,dict):continue
        if (region.get('origin')!='native' or region.get('task')!='text'
            or region.get('label') in {'image','header_image','footer_image'}
            or not box_ok(region.get('box')) or not isinstance(region.get('key'),str)
            or not 1<=len(region['key'])<=128):
            continue
        text=region.get('text')
        if not isinstance(text,str) or len(text)>MAX_TEXT:continue
        if region['key'] in seen:limited=True;continue
        seen.add(region['key'])
        problems=[]
        for o in observations(text):
            if not o['shapeMatches']:
                problems.append({**o,'reason':'labelled-literal-format-disagreement'})
        tokens=list(OPAQUE32.finditer(text))
        if len(tokens)>MAX_OPAQUE_TOKENS:
            limited=True;tokens=[]
        for m in tokens:
            raw=m[0];token=raw.upper()
            if token in anchors or not anchors:continue
            distances=[(sum(a!=b for a,b in zip(token,x)),x) for x in anchors]
            distance=min(n for n,_ in distances)
            # At most three differing glyphs, with one unambiguous closest anchor.
            if 1<=distance<=3 and sum(n==distance for n,_ in distances)==1:
                problems.append({'kind':'same-page-opaque-identifier-disagreement','literal':raw,
                    'start':m.start(),'end':m.end(),'reason':'different-nearby-typed-reading',
                    'distance':distance})
        if not problems:continue
        if len(requests)>=MAX_EXTRA_READS:limited=True;continue
        requests.append({'key':region['key'],'sourceBox':deepcopy(region['box']),
                         'strategy':'complementary-sdk-tight','problems':problems})
    return {'requests':requests,'limited':limited,'hashAnchorCount':len(anchors),
            'noTextCorrection':True,'noPreferredReading':True}
