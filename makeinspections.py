import os
import makeinspectionpage
from numpy import ceil

def makeinspections(inspectionsubname,objects,field):

    os.system('mkdir '+inspectionsubname)

    webfile=open(inspectionsubname+'/objects.html','w')
    webfile.write('<!DOCTYPE html>\n')
    webfile.write('<html>\n')
    webfile.write('<head>\n')
    webfile.write('<title>Clump Inspection for HALO7D objects</title>\n')

    webfile.write('<body>\n')

    webfile.write('<table cellspacing="10">\n')
    webfile.write('\t<tr>\n')

    for i in range(int(len(objects)/5)+1):
        
        if i==(int(len(objects)/5)):
            for jj in range(len(objects)%5):
                webfile.write('\t\t<td><a href=sites/inspectionsite_'+str(objects[int(5*i)+jj])+'.html>'+str(objects[int(5*i)+jj])+'</a></td>\n')
            webfile.write('\t</tr>\n')
        else:
            for jj in range(5):
                webfile.write('\t\t<td><a href=sites/inspectionsite_'+str(objects[int(5*i)+jj])+'.html>'+str(objects[int(5*i)+jj])+'</a></td>\n')
            webfile.write('\t</tr>\n')

    webfile.write('</table>\n')
    webfile.write('<a style="font-size:15px;text-align:center" href=example_issues.html>Example Issues</a>\n')
    webfile.write('</body>\n')
    webfile.write('</html>\n')
    webfile.close()

    os.system('mkdir '+inspectionsubname+'/cutouts')
    os.system('mkdir '+inspectionsubname+'/sites')
    os.system('mkdir '+inspectionsubname+'/qualities')
    os.system('cp redmarker.png '+inspectionsubname+'/sites')
    os.system('cp bluemarker.png '+inspectionsubname+'/sites')
    os.system('cp example_issues.html '+inspectionsubname)
    os.system('cp -r example_issues '+inspectionsubname)

    for i in range(len(objects)):
        if field=='goodss':
            try:
                if len(os.listdir('/Volumes/FantomHD/halo7d_data/specbyid_goodss/'+str(objects[i])+'/2D'))>0:
                    makeinspectionpage.makeinspectionsite(objects[i],field,prefix='/Users/carletont/halo7d/'+inspectionsubname,fileprefix='sites/inspectionsite_',relativedir='../cutouts/')
            except FileNotFoundError:
                None
        elif field=='goodsn':
            try:
                if len(os.listdir('/Volumes/FantomHD/halo7d_data/specbyid_goodsn/'+str(objects[i])+'/2D'))>0:
                    makeinspectionpage.makeinspectionsite(objects[i],field,prefix='/Users/carletont/halo7d/'+inspectionsubname,fileprefix='sites/inspectionsite_',relativedir='../cutouts/')
            except FileNotFoundError:
                None
        elif field=='cosmos':
            try:
                if len(os.listdir('/Volumes/FantomHD/halo7d_data/specbyid_cosmos/'+str(objects[i])+'/2D'))>0:
                    makeinspectionpage.makeinspectionsite(objects[i],field,prefix='/Users/carletont/halo7d/'+inspectionsubname,fileprefix='sites/inspectionsite_',relativedir='../cutouts/')
            except FileNotFoundError:
                None
        elif field=='egs':
            try:
                if len(os.listdir('/Volumes/FantomHD/halo7d_data/specbyid_egs/'+str(objects[i])+'/2D'))>0:
                    makeinspectionpage.makeinspectionsite(objects[i],field,prefix='/Users/carletont/halo7d/'+inspectionsubname,fileprefix='sites/inspectionsite_',relativedir='../cutouts/')
            except:
                None
        elif field=='uds':
            try:
                if len(os.listdir('/Volumes/FantomHD/halo7d_data/specbyid_uds/'+str(objects[i])+'/2D'))>0:
                    makeinspectionpage.makeinspectionsite(objects[i],field,prefix='/Users/carletont/halo7d/'+inspectionsubname,fileprefix='sites/inspectionsite_',relativedir='../cutouts/')
            except:
                None

