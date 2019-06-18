__author__ = 'alibaba'
__date__ = '2019/1/1'
import os
import glob
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def png2pdf(file_name):
    filename_pdf = file_name[:-4] + '.pdf'
    c = canvas.Canvas(filename_pdf, pagesize=landscape(A4))
    (w, h) = landscape(A4)
    c.drawImage(file_name, 0, 0, w, h)
    c.showPage()
    c.save()


path = os.path.abspath(os.path.join(os.getcwd(), __file__))
path = os.path.dirname(path)


file_list = glob.glob(os.path.join(path, '*.png'))
for filename in file_list:
    png2pdf(filename)

