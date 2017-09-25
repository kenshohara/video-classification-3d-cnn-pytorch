def classify_video(video_dir, video_name, class_names):
    
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = Variable(inputs, volatile=True)
        outputs = model(inputs)
