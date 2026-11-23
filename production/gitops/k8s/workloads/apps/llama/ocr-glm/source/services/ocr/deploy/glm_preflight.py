"""Offline identity check for the separate native GLM API; Paddle is unchanged."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from pathlib import Path, PurePosixPath

from preflight import PreflightError, keys, require, read_manifest, sha256_file, PYTHON_VERSION
from source_inventory import inventory as source_inventory, ROOT as SOURCE_ROOT
from verified_artifacts import artifact_identity, prime_cache, verify_cache
sys.path.insert(0,str(SOURCE_ROOT))
from services.ocr.glm.cpu_reader import reader_identity, public_reader_identity, verify_cpu_models

HERE=Path(__file__).resolve().parent
RUNTIME=Path('/opt/ocr-glm/runtime')
NATIVE=Path('/opt/llama/llama-server')
NATIVE_SHA='99183726b541dd7504a9fea2539d644b37bf0635d01a533f7366d003bd07b213'
ENGINE_VERSION='11277-eae11d221'
MODEL_ID='ggml-org/GLM-OCR-GGUF'
MODEL_REVISION='65a42de1148dbed2297e922b5dbc7d9b70c36578'
LAYOUT_ID='PaddlePaddle/PP-DocLayoutV3_safetensors'
LAYOUT_REVISION='e4f489dc536556fc1be02b973ba06756e0d4a2ac'
MODEL_FILES={
 'GLM-OCR-f16.gguf':'b06675e983db9593db78603b06f097e48c0cf078b37731c0a09612f4a249cf6f',
 'mmproj-GLM-OCR-Q8_0.gguf':'9c4b58e33e316ed142eb5dcb41abec3844d3e6e5dc361ffb782c3fa9d175141f',
 'model-manifest.json':'dba833a17034e6b62f83a68d6b72cec6235c79e6bd31dfa51f7fa6cd49852681',
}
LAYOUT_FILES={
 'config.json':'3cf834b91d23a756b1519bce4db42c09e852f3e35c35092dd5a3e253a50c071a',
 'preprocessor_config.json':'519fe0187a43a1ca429e3ad8317bab8700f0d5e8fb3a6e3a0a413ffac078ba42',
 'model.safetensors':'5ea422c6cc5fe759a47e1357c35639b58173508e025a3131cbe4b6ac59e2b85e',
}
WORKER_CONFIG={'configVersion':1,'layoutDevice':'cpu',
 'layoutModelDir':str(RUNTIME/'models/PP-DocLayoutV3_safetensors'),
 'nativeBaseUrl':'http://127.0.0.1:8001/v1','servedModelName':'glm-ocr','pdfScale':2}
AUX_MODEL_DIR='/models/ocr-aux/ppocr-v5-latin-v1'
INFERENCE={'engineVersion':ENGINE_VERSION,'binarySha256':NATIVE_SHA,'contextTokens':24576,
 'generationTokens':4096,'parallel':1,'threads':2,'imageMaxTokens':4096,
 'temperature':0,'topP':1,'seed':0,'metrics':True,'contextShift':False}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()


def tree_inventory(root):
 require(root.is_dir() and not root.is_symlink(),'Missing regular GLM runtime directory')
 result={}
 for path in sorted(root.rglob('*')):
  relative=path.relative_to(root).as_posix()
  if path.is_symlink():result[relative]={'link':str(path.readlink())}
  elif path.is_file():result[relative]={'sha256':sha256_file(path)}
  else:require(path.is_dir(),'Runtime contains a non-regular file')
 require(bool(result),'Empty GLM runtime')
 return result


def check_packages():
 require(platform.system()=='Linux' and platform.machine()=='x86_64','Requires Linux x86_64')
 require(platform.python_version()==PYTHON_VERSION,'Python patch does not match the pinned image')
 seen=set()
 for line in (HERE/'requirements.glm.lock').read_text().splitlines():
  match=re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+) --hash=sha256:[a-f0-9]{64}',line)
  require(match is not None,'Unpinned GLM dependency')
  name,version=match.groups();require(name not in seen,'Duplicate GLM dependency');seen.add(name)
  try:actual=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:raise PreflightError('Missing locked GLM dependency') from None
  require(actual==version,'GLM dependency differs from lock')
 require(len(seen)>=50,'Incomplete GLM dependency lock')


def validate_manifest(manifest,verify_files=True):
 expected_keys={'manifestVersion','provider','platform','pythonVersion','normalizationRevision','models','artifacts','inference','workerConfig'}
 if isinstance(manifest,dict) and 'auxiliaryReader' in manifest:expected_keys.add('auxiliaryReader')
 keys(manifest,expected_keys,'GLM manifest')
 require(type(manifest['manifestVersion']) is int and manifest['manifestVersion']==1,'Unsupported GLM manifest')
 require(manifest['provider']=='glm' and manifest['platform']=='linux/amd64' and manifest['pythonVersion']==PYTHON_VERSION,'Unexpected GLM runtime')
 require(manifest['normalizationRevision']=='receipt-core-ptbr-v28','Unexpected GLM normalization rules')
 require(manifest['inference']==INFERENCE and manifest['workerConfig']==WORKER_CONFIG,'Unreviewed GLM inference settings')
 if 'auxiliaryReader' in manifest:
  require(canonical(manifest['auxiliaryReader'])==canonical({'modelDir':AUX_MODEL_DIR,'identity':reader_identity()}),'Auxiliary reader identity or settings differ')
 artifacts=manifest['artifacts'];keys(artifacts,{'image','adapterTreeSha256','sdkLockSha256','runtimeTreeSha256'},'GLM artifacts')
 require(isinstance(artifacts['image'],str) and re.fullmatch(r'[a-z0-9./:_-]+@sha256:[a-f0-9]{64}',artifacts['image']) is not None,'Missing immutable GLM image')
 for key in ['adapterTreeSha256','sdkLockSha256','runtimeTreeSha256']:
  require(isinstance(artifacts[key],str) and re.fullmatch('[a-f0-9]{64}',artifacts[key]) is not None,'Invalid GLM artifact digest')
 require(artifacts['adapterTreeSha256']==source_inventory()['sha256'],'GLM source differs from manifest')
 require(artifacts['sdkLockSha256']==sha256_file(HERE/'requirements.glm.lock'),'GLM dependency lock differs')
 keys(manifest['models'],{'ocr','layout'},'GLM models')
 for name,model_id,revision,directory,files in [
  ('ocr',MODEL_ID,MODEL_REVISION,'/models/glm-ocr-baseline',MODEL_FILES),
  ('layout',LAYOUT_ID,LAYOUT_REVISION,WORKER_CONFIG['layoutModelDir'],LAYOUT_FILES)]:
  model=manifest['models'][name];keys(model,{'id','revision','path','files'},'GLM model')
  require(model=={'id':model_id,'revision':revision,'path':directory,'files':files},'GLM model identity or bytes differ')
  if verify_files:
   base=Path(directory);require(base.is_dir() and not base.is_symlink(),'Missing GLM model directory')
   actual={}
   for path in base.rglob('*'):
    require(not path.is_symlink(),'GLM model contains a symbolic link')
    if path.is_dir():continue
    require(path.is_file(),'GLM model contains a non-regular file')
    actual[path.relative_to(base).as_posix()]=sha256_file(path)
   require(actual==files,'GLM model inventory differs')
 if verify_files:
  require(sha256_file(NATIVE)==NATIVE_SHA,'Native GLM executable differs')
  require(hashlib.sha256(canonical(tree_inventory(RUNTIME))).hexdigest()==artifacts['runtimeTreeSha256'],'GLM runtime tree differs')
  verify_auxiliary(manifest)
 return manifest


def verify_auxiliary(manifest):
 if 'auxiliaryReader' not in manifest:return
 try:verify_cpu_models(AUX_MODEL_DIR,time.monotonic()+50)
 except Exception:raise PreflightError('Auxiliary reader offline verification failed') from None


def runtime_config(manifest):
 fingerprint=hashlib.sha256(canonical(manifest)).hexdigest()
 result={'pipeline':{'id':'glm-ocr-'+fingerprint[:16],'fingerprint':fingerprint,'executionMode':'real',
  'ocrModel':{'id':MODEL_ID,'revision':MODEL_REVISION},'layoutModel':{'id':LAYOUT_ID,'revision':LAYOUT_REVISION},
  'interpretationModel':None,'inferenceEngine':{'name':'llama-cpp','version':ENGINE_VERSION}},'workerConfig':dict(WORKER_CONFIG)}
 if 'auxiliaryReader' in manifest:
  reader=public_reader_identity()
  result['pipeline']['auxiliaryReader']=reader
  result['workerConfig']['cpuReader']={'id':reader['id'],'fingerprint':reader['fingerprint'],'modelDir':AUX_MODEL_DIR}
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--emit-runtime',action='store_true')
 modes=parser.add_mutually_exclusive_group()
 modes.add_argument('--verified-cache',type=Path)
 modes.add_argument('--prime-cache',type=Path)
 args=parser.parse_args()
 try:
  check_packages();manifest=validate_manifest(read_manifest(args.manifest),verify_files=False)
  record=json.loads(Path('/opt/ocr-glm/source-tree.json').read_text())
  require(record==source_inventory(),'GLM source differs from built image')
  roots={'runtime':RUNTIME,'ocr':Path(manifest['models']['ocr']['path'])}
  if args.verified_cache:
   verify_cache(args.verified_cache,artifact_identity(manifest),roots)
   require(sha256_file(NATIVE)==NATIVE_SHA,'Native GLM executable differs')
   verify_auxiliary(manifest)
  elif args.prime_cache:
   prime_cache(args.prime_cache,artifact_identity(manifest),roots,lambda:validate_manifest(manifest))
  else:validate_manifest(manifest)
  require(record==source_inventory(),'GLM source changed during preflight')
  if args.emit_runtime:print(json.dumps(runtime_config(manifest),separators=(',',':')))
 except (PreflightError,OSError,ValueError,TypeError):
  print('GLM OCR preflight rejected startup; check the pinned manifest and runtime.',file=sys.stderr)
  return 2
 return 0


if __name__=='__main__':raise SystemExit(main())
