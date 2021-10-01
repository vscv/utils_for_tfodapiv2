# utils_for_tfodapiv2
Utils/Tools for



* * *
## What can this utils do?
This 100% Glue Code is for procesing image/annotation/visualizaiton/traing_set/CSV when working on Tensorflow object detection API v2.



* * *
## Show coco json's label/bbox
 注意！輸入或返回都是array型態，輸入[ids]，返回[{}, {}, {}...]。
 尤其輸入ids剛好是數值如id:123，api會切成[1,2,3]，而返回了imgIds=1,2,and 3的anns。

    annotations_file='/PATH_TO/train.json'
    coco=COCO(annotations_file)

    #check length of coco ImgIds
    imgIds = coco.getImgIds()#A list of id
    # print(f'imgIds: {imgIds}')
    print("[Total imgIds]: ", len(imgIds))

    ## user give ids
    give_id = 111 # str(300) # or '0'
    print(f'[give_id]: {give_id}')

    # img_info = coco.loadImgs(give_id)[0]
    img_infos = coco.loadImgs(imgIds)#A list of {'id': , 'width': , 'height': , 'license': , 'file_name': }
    print(f'[img_infos[:5]]: {img_infos[:5]}\n')


    img_info = img_infos[give_id]
    print(f'[img_info]: {img_info}\n')

    img_name = img_info['file_name']
    print(f'[image name]: {img_name}')

    img_id = img_info['id']
    print(f'[img_id]: {img_id}')



    # ann=coco.loadAnns(coco.getAnnIds(img_id))
    # ann_ids=coco.getAnnIds('image_00000')
    # print(f'\n[ann_ids[:11]]: {ann_ids[:]}')

    ann=coco.loadAnns(coco.getAnnIds([img_id]))#imgIds=img_info['id']
    print(f'[# of annotations in {img_name}]: {len(ann)} ')
    # print(f'[ann]: {ann}')


    img =  cv2.imread("/PATH_TO/" + img_name)

    #To show patchs and points  
    fig = plt.figure()         
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')

    coco.showAnns(ann, draw_bbox=True)




* * *
## Credits
Most of this code was modified or inspired from the great work of GitHub, StackOverflow etc.
