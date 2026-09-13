"""Build the course paper and appendix from audited evidence.

Run from repository root: python production/build_course_report.py
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
from pypdf import PdfReader
from reports.article_layout import Article
from production.course_evidence import *
from production.course_figures import prepare_figures
from production.course_manuscript import write_main,write_appendix

def build():
    check_frozen()
    prepare_figures()
    records={}
    for filename,title,writer,expected in [
        ('8_pages_draft.pdf','Compact Generative Models of Human Interception Movements',write_main,8),
        ('Course_Report_Appendix.pdf','Supplementary Methods and Results',write_appendix,10)]:
        a=Article(OUT/filename,title,'Seman Libbiss and Paz Flashner<br/>Workshop on Deep Learning, Tel Aviv University<br/>Research supervision: Jason Friedman; course advisor: Moni Shahar')
        a.styles['body'].fontSize=10.2;a.styles['body'].leading=12.5;a.styles['body'].spaceAfter=5
        a.styles['heading'].spaceBefore=9;a.styles['heading'].spaceAfter=5
        a.styles['caption'].fontSize=8.7;a.styles['caption'].leading=10.5;a.styles['caption'].spaceAfter=6
        a.styles['table'].fontSize=8.2;a.styles['table'].leading=9.7
        writer(a);a.finish()
        pages=PdfReader(OUT/filename).pages
        records[filename]={'pages':len(pages),'page_text_lengths':[len(p.extract_text()) for p in pages]}
        print(filename,len(pages),records[filename]['page_text_lengths'])
        assert len(pages)==expected,(filename,len(pages),expected)
    check_frozen()
    (OUT/'COURSE_REPORT_VERIFICATION.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':build()
