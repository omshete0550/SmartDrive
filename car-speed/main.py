import car_speed.car_speed_detection as car_speed

car_speed.read('train.mp4', 'Car_Detection_images/')
car_speed.preprocess('Car_Detection_images', 'train.txt', 'feature.txt', resize = 0.5, x_slice = 8, y_slice = 6)
mse, MEAN_CONST, STD_CONST = car_speed.train('feature.txt')
car_speed.speed_detection('Model.h5', 'test.mp4', 'detect_result.txt', 0.5, 8, 6, MEAN_CONST, STD_CONST)
car_speed.combine_video_and_speed('test.mp4', 'detect_result.txt', 'test_with_speed.mp4')