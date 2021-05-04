import numpy as np
import time
import cv2
import os

DEBUG = False

#INPUT_FILE='data/validate/idiots.mp4'
INPUT_FOLDER='extract/input'
OUTPUT_FOLDER='extract/output'
LABELS_FILE='data/obj.names'
CONFIG_FILE='yolov4-tiny-custom.cfg'
WEIGHTS_FILE='yolov4-tiny-custom_6000.weights'

MIN_WIDTH = 50
MIN_HEIGHT = 20
CONFIDENCE_THRESHOLD=0.5

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

imgnames = os.listdir(INPUT_FOLDER)
    


for imgname in imgnames:        
	pre, ext = os.path.splitext(imgname)
	imgpath = os.path.join(INPUT_FOLDER,imgname)
	image = cv2.imread(imgpath)
	(H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	if DEBUG:
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))


	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE_THRESHOLD:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
		CONFIDENCE_THRESHOLD)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			if w < MIN_WIDTH or h < MIN_HEIGHT:
				continue

			color = (138,43,226)
			ROI = image[y:y+h, x:x+w]
			
			#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			#	0.5, color, 2)
			output_name = pre + "_" + str(i) + ".jpg"
			output_path = os.path.join(OUTPUT_FOLDER,output_name)
			try:
				cv2.imwrite(output_path,ROI)
			except Exception as e:
				print(e)

	# show the output image
	#out.write(image)
	if DEBUG:
		cv2.imshow("window", ROI)
		key = cv2.waitKey()
		if key == 27:#if ESC is pressed, exit loop
			cv2.destroyAllWindows()
			break

