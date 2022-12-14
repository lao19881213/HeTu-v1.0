import os
import os.path as osp
import json
import xml.etree.ElementTree as ET

from random import sample

import sqlite3 as dbdrv

from astropy.io import fits
from astropy import wcs

from utils.viz_result import show_n_save_img

"""
Prepare RGZ data into the COCO format based on
http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

TODO - add host galaxy as a keypoint as done in the mask-rcnn paper
"""

IMG_SIZE = 132
NUM_CLASSES = 4 # or 6

#CAT_XML_COCO_DICT = {'1C': 1, '2C': 2, '3C': 3, '1_1': 1, '1_2': 2, '1_3': 3,
#                     '2_2': 4, '2_3': 5, '3_3': 6}

if (6 == NUM_CLASSES):
   CAT_XML_COCO_DICT = {'1_1': 1, '1_2': 2, '1_3': 3,
                       '2_2': 4, '2_3': 5, '3_3': 6}
elif (4 == NUM_CLASSES):
   CAT_XML_COCO_DICT = {'cs': 1, 'fr1': 2, 'fr2': 3,
                       'core_jet': 4}
else:
   raise Exception('Incorrect NUM_CLASSES')

def create_categories():
    catlist = []
    if (3 == NUM_CLASSES):
        catlist.append({"supercategory": "galaxy", "id": 1, "name": "1C"})
        catlist.append({"supercategory": "galaxy", "id": 2, "name": "2C"})
        catlist.append({"supercategory": "galaxy", "id": 3, "name": "3C"})
    elif (6 == NUM_CLASSES):
        catlist.append({"supercategory": "galaxy", "id": 1, "name": "1C_1P"})
        catlist.append({"supercategory": "galaxy", "id": 2, "name": "1C_2P"})
        catlist.append({"supercategory": "galaxy", "id": 3, "name": "1C_3P"})
        catlist.append({"supercategory": "galaxy", "id": 4, "name": "2C_2P"})
        catlist.append({"supercategory": "galaxy", "id": 5, "name": "2C_3P"})
        catlist.append({"supercategory": "galaxy", "id": 6, "name": "3C_3P"})
    elif (4 == NUM_CLASSES):
        catlist.append({"supercategory": "galaxy", "id": 1, "name": "cs"})
        catlist.append({"supercategory": "galaxy", "id": 2, "name": "fr1"})
        catlist.append({"supercategory": "galaxy", "id": 3, "name": "fr2"})
        catlist.append({"supercategory": "galaxy", "id": 4, "name": "core_jet"})
    else:
        raise Exception('Incorrect NUM_CLASSES')
    return catlist

def create_coco_anno():
    anno = dict()
    anno['info'] = {"description": "RGZ data release 1", "year": 2018}
    anno['licenses'] = [{"url": r"http://creativecommons.org/licenses/by-nc-sa/2.0/", 
                         "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}]
    anno['images'] = []
    anno['annotations'] = []
    anno['categories'] = create_categories()
    return anno

def _get_xml_metadata(img_id, xml_file, start_anno_id):
    ret = dict()
    tree = ET.parse(xml_file)
    ret['width'] = int(tree.find('size').find('width').text)
    ret['height'] = int(tree.find('size').find('height').text)
    objs = tree.findall('object')
    anno_list = []
    for idx, obj in enumerate(objs):
        anno = dict()
        anno['category_id'] = CAT_XML_COCO_DICT[obj.find('name').text]
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        bw = x2 - x1
        bh = y2 - y1
        anno['bbox'] = [x1, y1, bw, bh]
        anno['area'] = bh * bw #TODO mask will be different than this
        anno['id'] = start_anno_id + idx
        anno['image_id'] = img_id
        anno['iscrowd'] = 0
        anno_list.append(anno)
    ret['num_objs'] = len(objs)
    ret['anno_list'] = anno_list
    return ret

def xml2coco(img_list_file, in_img_dir, xml_dir, out_img_dir, json_dir):
    """
    convert the "old" rgz_rcnn format (xml) to HeTu coco format

    img_list_file:   a text file with a list of image names w/o file extensions (e.g. ".png")
    """
    anno = create_coco_anno()
    images = anno['images']
    with open(img_list_file, 'r') as fin:
        imglist = fin.read().splitlines()
    start_anno_id = 0
    for idx, img in enumerate(imglist):
        img_d = {'id': idx, 'license': 1, 'file_name': '%s.png' % img}
        xml_file = os.path.join(xml_dir, '%s.xml' % img)
        xml_meta = _get_xml_metadata(idx, xml_file, start_anno_id)
        start_anno_id += xml_meta['num_objs']
        img_d['height'], img_d['width'] = xml_meta['height'], xml_meta['width']
        images.append(img_d)
        anno['annotations'].extend(xml_meta['anno_list'])
        if (idx % 300 == 0 and idx > 0):
            print("Processed %d xml files" % idx)
    json_dump = osp.join(json_dir, osp.splitext(osp.basename(img_list_file))[0] + '.json')
    with open(json_dump, 'w') as fout:
        json.dump(anno, fout)

sql_str = """
    select a.cls_lbl, b.xmin, b.ymin, b.xmax, b.ymax, 
    c.host_ra, c.host_dec from rgz_samples a, anno b, host_gal c where 
    a.first_id = '%s' and 
    a.catalog_id = b.catalog_id and 
    a.catalog_id = c.catalog_id
    """

def _get_obj_metadata(rslist, start_anno_id, img_id, wcoord, width, height):
    """
    rslist - db resultSet list
    """
    ret = dict()
    anno_list = []
    for idx, rs in enumerate(rslist):
        anno = dict()
        anno['category_id'] = CAT_XML_COCO_DICT[rs[0]]
        x1, y1, x2, y2 = rs[1:5]
        bw = x2 - x1
        bh = y2 - y1
        anno['bbox'] = [x1, y1, bw, bh]
        anno['area'] = bh * bw #TODO mask will be different than this
        anno['id'] = start_anno_id + idx
        anno['image_id'] = img_id
        anno['iscrowd'] = 0
        if (wcoord is not None):
            host_ra, host_dec = float(rs[5]), float(rs[6])
            if (-99 in [host_ra, host_dec]):
                host_x = (x1 + x2) / 2
                host_y = (y1 + y2) / 2
            else:
                host_x, host_y = wcoord.wcs_world2pix(host_ra, host_dec, 0)
                host_y = height - host_y # FITS pixel (0,0) starts from bottom left
            hx, hy = int(host_x), int(host_y)
            hx1 = max(hx - 1, 0)
            hy1 = max(hy - 1, 0)
            hx2 = min(hx + 1, width - 1)
            hy2 = min(hy + 1, height - 1)
            anno['segmentation'] = [[hx1, hy1, hx2, hy1, hx2, hy2, hx1, hy2]]

        anno_list.append(anno)
    ret['num_objs'] = len(rslist)
    ret['anno_list'] = anno_list
    return ret

def db2coco(db_path, img_list_file, fits_dir, json_dir):
    """
    convert the sqlite db record into HeTu coco format

    img_list_file:   a text file with a list of image names w/o file extensions (e.g. ".fits")
    """
    def query_db(dbconn, query, read_only=True):
        cur = dbconn.cursor()
        cur.execute(query)
        if (read_only):
            rslist = cur.fetchall()  # not OK if we have millions of records
            cur.close()
            return rslist
        else:
            dbconn.commit()

    anno = create_coco_anno()
    images = anno['images']

    with open(img_list_file, 'r') as fin:
        imglist = fin.read().splitlines()
    
    conn = dbdrv.connect(db_path)
    start_anno_id = 0
    for idx, img in enumerate(imglist):
        img_d = {'id': idx, 'license': 1, 'file_name': '%s.png' % img}
        first_id = img.split('_')[0]
        fits_fn = osp.join(fits_dir, '%s.fits' % first_id)
        hdulist = fits.open(fits_fn)
        w = wcs.WCS(hdulist[0].header)
        height, width = hdulist[0].data.shape[0:2]
        query = sql_str % first_id
        rslist = query_db(conn, query)
        obj_meta = _get_obj_metadata(rslist, start_anno_id, idx, w, width, height)
        start_anno_id += obj_meta['num_objs']
        img_d['height'], img_d['width'] = height, width
        images.append(img_d)
        anno['annotations'].extend(obj_meta['anno_list'])
        if (idx % 50 == 0 and idx > 0):
            print("Processed %d FITS files" % idx)
    json_dump = osp.join(json_dir, osp.splitext(osp.basename(img_list_file))[0] + '_hg.json')
    with open(json_dump, 'w') as fout:
        json.dump(anno, fout)

def check_hg(json_file, num_viz=50):
    """
    randomly check the correctness of the gt host galaxy position
    """
    from dataset import COCODetection
    cocod = COCODetection('data', 'trainD3_hg')
    with open(json_file, 'r') as fin:
        anno = json.load(fin)
    images = anno['images']
    images = sample(images, num_viz)
    for image in images:
        png_file = osp.join(cocod._imgdir, image['file_name'])
        objs = cocod.coco.imgToAnns[image['id']]
        boxes = []
        segs = []
        classes = []
        for objid, obj in enumerate(objs):
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = obj['bbox']
            x2 = x1 + w
            y2 = y1 + h
            boxes.append((x1, y1, x2, y2))
            points8 = obj['segmentation'][0]
            # [[hx1, hy1, hx2, hy1, hx2, hy2, hx1, hy2]]
            xs1, ys1, xs2, ys2 = points8[0], points8[1], points8[4], points8[5]
            segs.append((xs1, ys1, xs2, ys2))
            classes.append(obj['category_id'])
        out_fn = osp.join('/tmp', image['file_name'])
        show_n_save_img(png_file, boxes, segs, out_fn, classes, 
                        image['height'], image['width'])

if __name__ == '__main__':
    img_list_file = '/Users/chen/gitrepos/ml/' +\
                    'rgz_rcnn/data/RGZdevkit2017/RGZ2017/ImageSets/Main/testD1.txt'
    in_img_dir = None
    xml_dir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/RGZ2017/Annotations'
    out_img_dir = None
    json_dir = '.'
    #xml2coco(img_list_file, in_img_dir, xml_dir, out_img_dir, json_dir)
    db_path = '/Users/chen/gitrepos/ml/rgz-ml/data/rgz_image_set.sqlite3'
    fits_dir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/RGZ2017/FITSImages'
    #db2coco(db_path, img_list_file, fits_dir, json_dir)
    json_file = 'data/annotations/instances_trainD3_hg.json'
    check_hg(json_file)


        
