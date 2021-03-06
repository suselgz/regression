import cv2
import numpy as np
minThres = 20
def show(img,name="img"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Excuet():
    # 读取图像1
    img=cv2.imread('./images/3.jpg')
    if img is None:
        return
    img=~img
    img2=img.copy()
    img2=~img2
    show(img, "img")
    img = cv2.medianBlur(img, 5)
    show(img, "medianBlur")
    return
  #  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #  _, thres = cv2.threshold(gray, minThres, 255, cv2.THRESH_BINARY)
  #  show(thres, "thres")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 椭圆结构
    dilation = cv2.dilate(img, kernel)  # 膨胀
    show(dilation, "dilation")

    edges = cv2.Canny(dilation, 10, 200)
    show(edges, "edges")
    # 图像差分
    diff = cv2.absdiff(img, img2)
  #  show(diff,"diff") # 结果图
  #  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))# 椭圆结构
  #  erosion = cv2.erode(img, kernel)
  #  show(erosion,"erosion")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
    dilation = cv2.dilate(img, kernel)  # 膨胀
  #  show(dilation, "dilation")


    gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(gray, minThres, 255, cv2.THRESH_BINARY)
  #  show(thres, "thres")  # 结果图
if __name__ == "__main__":
    Excuet()