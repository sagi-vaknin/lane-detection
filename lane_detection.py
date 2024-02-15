import numpy as np
import cv2

def region_selection(image,mode):
	#generate blank frame
	mask = np.zeros_like(image) 

	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	rows, _ = image.shape[:2]
	#region of interest polygon vertices based on video mode
	if mode:
		vertices = np.array([[(700, rows-100),
        (1200,1100),
        (1700,1100),
        (2200, rows-100)]], dtype=np.int32)
	else:
		vertices = np.array([[(1000, rows-100),
        (1350,1200),
        (1550,1200),
        (1950, rows-100)]], dtype=np.int32)
	#generate filled polygon
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	#cut only the relevant parts of the image
	masked_image = cv2.bitwise_and(image, mask)
	
	return masked_image

def display_lane_change_text(image, direction):
	image_height, image_width = image.shape[:2]
	#generate blank frame
	text_image = np.zeros_like(image)

	#generate text based on the direction of change
	if direction == "right":
		text = "Lane Change ->"
	elif direction == "left":
		text = "Lane Change <-"
	else:
		text = "Lane Change"
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 8
	font_thickness = 20
	text_color = (0, 0, 255)  #red text

	#generate text size
	text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

	#generate text location
	text_x = (image_width - text_size[0]) // 2
	text_y = text_size[1] + 200  

	#add the text to the frame
	cv2.putText(text_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
	return text_image

def adjust_brightness(image, factor):
	#turn image into HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
	# add factor to each pixel in v to increase brightness
    v = cv2.add(v, factor)
    v = np.clip(v, 0, 255)
	#merge altered v back to the image
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, factor):
	#turn image into LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
	#add factor to each pixel in L to increate contrast
    l = cv2.add(l, factor)
    l = np.clip(l, 0, 255)
	#merge altered L to the image
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def histogram_equalization(image):
	#turn image into HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
	#perform histogram equalization to the v channel
    v = cv2.equalizeHist(v)
	#merge altered v to the image
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def hough_transform(image):
	return cv2.HoughLinesP(image, rho = 1, theta =  np.pi/180, threshold = 20, minLineLength = 20, maxLineGap = 400)
	
def average_slope_intercept(lines):
	left_lines, right_lines = [], []
	left_weights, right_weights = [], []
	
	#iterate on every line
	for line in lines:
		for x1, y1, x2, y2 in line:
			#skipping vertical lines to avoid division by zero
			if x1 == x2:
				continue
			#calculating slope of a line
			slope = (y2 - y1) / (x2 - x1)
			#calculating intercept of a line
			intercept = y1 - (slope * x1)
			#calculating length of a line
			length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
			#slope of left lane is negative and for right lane slope is positive
			if slope < 0:
				left_lines.append((slope, intercept))
				left_weights.append((length))
			else:
				right_lines.append((slope, intercept))
				right_weights.append((length))
	#weighted average to find best fitting lanes
	left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
	right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
	return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    
    slope, intercept = line

    #check if slope is smaller than epsilon to avoid division by zero
    if abs(slope) < 1e-6:
        x1 = x2 = int(intercept)
    else:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
	left_lane, right_lane = average_slope_intercept(lines)
	y1 = image.shape[0]
	y2 = y1 * 0.6 + 230
	left_line = pixel_points(y1, y2, left_lane)
	right_line = pixel_points(y1, y2, right_lane)
	return left_line, right_line

	
def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=25):
	line_image = np.zeros_like(image)
	#iterate on given lines and draw them on an empty image
	for line in lines:
		if line is not None:
			cv2.line(line_image, *line, color, thickness)
	#if there are 2 lines
	if len(lines) == 2 and None not in lines:
		x_left_top, y_left_top = lines[0][0]  #top of the left line
		x_right_top, y_right_top = lines[1][0]  #top of the right line
		x_left_bottom, y_left_bottom = lines[0][1]  #bottom of the left line
		x_right_bottom, y_right_bottom = lines[1][1]  #bottom point of the right line

		#extract relevant points to fill the entire lane with color
		pts = np.array([[x_left_top, y_left_top], [x_right_top, y_right_top], 
							[x_right_bottom, y_right_bottom], [x_left_bottom, y_left_bottom]], np.int32)
		pts = pts.reshape((-1, 1, 2))

		#fill the lane
		cv2.fillPoly(line_image, [pts], [0,0,255])

	#combine the identified lane with the original image
	result = cv2.addWeighted(image, 1.0, line_image, 0.5, 0.0)
	#if a lane line is missing -> there is a lane change
	if lines[0] == None:
		text_image = display_lane_change_text(image, "right")
		result = cv2.addWeighted(result, 1.0, text_image, 1, 0.0)
	elif lines[1] == None:
		text_image = display_lane_change_text(image, "left")
		result = cv2.addWeighted(result, 1.0, text_image, 1, 0.0)

	return result

def frame_processor(image):
	if mode:
		#make brightness and contrast changes to the inputted frame as a pre-processing stage
		brightness_adjusted = adjust_brightness(image, 30)  
		contrast_adjusted = adjust_contrast(brightness_adjusted, 50)  
		equalized_image = histogram_equalization(contrast_adjusted)
	else:
		#video is not night-mode, not changing the image
		equalized_image = image.copy()
	#turning frame into grayscale
	grayscale = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)
	#gaussian blur to reduce noise
	blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
	#thresholding and region selecting based on video mode
	if mode:
		_, binary_threshold = cv2.threshold(blur, 225, 250, cv2.THRESH_BINARY)
		region = region_selection(binary_threshold,True)
	else:
		_, binary_threshold = cv2.threshold(blur, 150, 190, cv2.THRESH_BINARY)
		region = region_selection(binary_threshold,False)
		
	#hough transform for line detection
	hough = hough_transform(region)
	#draw the lines on the input frame
	result = draw_lane_lines(image, lane_lines(image, hough))
	return result


#define input output properties etc.
#change the input video to be day.mp4 or night.mp4 as requested
video_path = "input_videos/day.mp4"
cap = cv2.VideoCapture(video_path)
mode = video_path == "input_videos/night.mp4"

if mode:
	output_file_path = "output_videos/night_output.mp4"
else:
	output_file_path = "output_videos/day_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_file_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
print("Generating output video...")
skip_frames = 3

#loop through each frame of the input video
while cap.isOpened():
	#skipping some frames to make the video faster 
	#(it became slow due to many calculations, so added this part)
	for _ in range(skip_frames):
		ret, frame = cap.read()
		if not ret:
			break

	if not ret:
		break
    
	#process the current given frame
	processed_frame = frame_processor(frame)

	#write new frame to the output video
	output_video.write(processed_frame)


cap.release()
output_video.release()
print("output video generated!")
cv2.destroyAllWindows()