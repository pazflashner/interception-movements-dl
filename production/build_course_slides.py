"""Build editable PPTX and its rendered PDF using the Codex artifact-tool runtime.

Ordinary report builds do not need Node. Set COURSE_RUNTIME_ROOT to override the
default bundled dependency directory on another computer.
"""
import os,sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reportlab.pdfgen import canvas
from production.course_evidence import OUT
from production.course_figures import prepare_figures
from production.course_slides import make_slides

def build():
    runtime=Path(os.environ.get('COURSE_RUNTIME_ROOT',Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies'))
    skill=Path(os.environ.get('COURSE_SLIDE_SKILL',Path.home()/'.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations'))
    node=runtime/'node/bin/node.exe'
    if not node.exists() or not skill.exists():
        raise RuntimeError('Presentation authoring needs the bundled artifact-tool runtime. Set COURSE_RUNTIME_ROOT and COURSE_SLIDE_SKILL, or edit the delivered PPTX directly in PowerPoint.')
    prepare_figures();slides=make_slides()
    env=os.environ.copy();env.update(COURSE_REPO=str(ROOT),COURSE_SLIDE_SKILL=str(skill),COURSE_RUNTIME_PYTHON=str(runtime/'python/python.exe'),RUNTIME_NODE_MODULES=str(runtime/'node/node_modules'))
    subprocess.run([str(node),str(OUT/'build_course_presentation.mjs')],env=env,cwd=ROOT,check=True)
    c=canvas.Canvas(str(OUT/'10_min_presentation.pdf'),pagesize=(960,540))
    c.setTitle('Interception Movement Distributions: Course Presentation')
    c.setAuthor('Seman Libbiss and Paz Flashner')
    for i,s in enumerate(slides,1):
        # These are exact slide renders; the companion PPTX retains native text/tables.
        c.drawImage(str(OUT/f'assets/course_slide_build/slide-{i:02}.png'),0,0,width=960,height=540)
        key=f'slide{i}';c.bookmarkPage(key);c.addOutlineEntry(f'{i}. {s["title"]}',key,level=0)
        c.showPage()
    c.save();print('25-page presentation PDF exported from the same slide renders.')

if __name__=='__main__':build()
