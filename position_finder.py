import cv2
import pickle

width = 107       # 156 - 49
height = 48       # 193 - 145

# save our previous task which was performed in last time and open the file from that part
try:
    with open('carparkpos', 'rb') as f:
        poslist = pickle.load(f)
except:
    poslist = []

def mouseClick(events, x, y, flags, params):
    # user press the left button of the mouse we will get the coordinates of those value
    if events == cv2.EVENT_LBUTTONDOWN:
        # x is width and y is height
        poslist.append((x,y))

    # user press the right button of the mouse then remove the rectangle from their
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(poslist):
            x1, y1 = pos
            if x1<x<x1+width and y1<y<y1+height:
                poslist.pop()

    with open('carparkpos','wb') as f:
        pickle.dump(poslist, f)

while True:
    image = cv2.imread("Image_Video/image.png")
    for pos in poslist:
        # starting point and ending point
        cv2.rectangle(image, (pos[0],pos[1]), (pos[0] + width, pos[1] + height), (255 ,100, 100),2)
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", mouseClick)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break