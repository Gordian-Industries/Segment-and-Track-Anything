sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.20,
        'stability_score_thresh': 0.20,
        'crop_n_layers':  1,
        'crop_n_points_downscale_factor': 4,
        'min_mask_region_area': 200,
    }

input_image = "./assets/cropped_image.jpg"
image = cv2.imread(input_image)
# cap = cv2.VideoCapture(io_args['input_video'])
# frame_idx = 0
segtracker = SegTracker(segtracker_args, sam_args, aot_args)
segtracker.restart_tracker()
with torch.cuda.amp.autocast():
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_mask, _ = segtracker.detect_and_seg(origin_frame=image, grounding_caption="get items", box_threshold=0.10,
                                             text_threshold=0.25)
    torch.cuda.empty_cache()
    obj_ids = np.unique(pred_mask)
    obj_ids = obj_ids[obj_ids != 0]

    init_res = draw_mask(image, pred_mask, id_countour=False)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(init_res)
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(colorize_mask(pred_mask))
    plt.show()

    del segtracker
    torch.cuda.empty_cache()
    gc.collect()