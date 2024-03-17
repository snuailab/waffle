# change `image`(cv2, BGR) as you want.
image = image[:, ::-1]

import cv2

image = cv2.blur(image, (15, 15))
