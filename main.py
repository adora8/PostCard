import cv2


# 定义轮廓排序函数
def sort_contours(cnts, method="left-to-right"):
    # 正序排列
    reverse = False
    # 左右排列
    i = 0
    # 倒序排列
    if method == "right-to-left" or "bottom-to-top":
        reverse = True
    # 上下排列
    if method == "top-to-bottom" or "bottom-to-top":
        i = 1
    # 用一个最小的矩形，把找到的图形包起来 x,y,w,h
    boundingboxes = [cv2.boundingRect(c) for c in cnts]
    # 对轮廓进行压缩，排序
    (cnts, boundingboxes) = zip(*(sorted(zip(cnts, boundingboxes), key=lambda b: b[i][1], reverse=reverse)))
    return cnts, boundingboxes


# 定义调整图像尺寸函数
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    if height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    # cv2.resize(原图src,输出尺寸dsize,插值方法interpolation[cv2.INTER_AREA基于局部像素的重采样])
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


# 定义输出图像函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 引入数字模板
img = cv2.imread('D:/360Downloads/wpcache/srvsetwp/number.jpg')
cv_show('template', img)
# 灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('template_gray', img_gray)
# 二值化
img_er = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('template_er', img_er)
# 计算轮廓并画出
# 最终得到的是img_er
refCnts = cv2.findContours(img_er.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(img_er, refCnts, -1, (0, 0, 255), 3)
cv_show('template_cnts', img_er)
# 将轮廓逐个输出
refCnts = sort_contours(refCnts, method="left-to-right")[0]
digits = {}
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = img_er[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 58))
    digits[i] = roi

import numpy as np

# 通过银行卡号第一个数字区分是哪种银行卡
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card",
    "9": "Error"
}
sqKernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
rectKernal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
# 引入银行卡图片
image = cv2.imread('D:/360Downloads/wpcache/srvsetwp/postcard.jpg')
cv_show('image', image)
# 调整银行卡图片大小
image = resize(image, width=300)
# 灰度图
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', image_gray)
# 礼帽操作
image_tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, rectKernal)
cv_show('tophat', image_tophat)
# Sobel算子 cv2.Sobel(src原始图像, ddepth输出图像深度, dx在x方向的求导阶数, dy在y方向的求导阶数, ksize代表Sobel核的大小)
image_gradX = cv2.Sobel(image_tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# 对数组中每个元素求绝对值
image_gradX = np.absolute(image_gradX)
# 归一化
(minVal, maxVal) = (np.min(image_gradX), np.max(image_gradX))
image_gradX = (255 * ((image_gradX - minVal) / (maxVal - minVal)))
image_gradX = image_gradX.astype('uint8')
print(np.array(image_gradX).shape)
cv_show('gradX', image_gradX)
# 闭操作
image_close = cv2.morphologyEx(image_gradX, cv2.MORPH_CLOSE, rectKernal)
cv_show('close', image_close)
# 图像阈值 cv2.threshold(src原始的灰度图像, thresh起始阈值, maxval最大阈值, type定义如何处理数据与阈值的关系)
image_thresh = cv2.threshold(image_close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', image_thresh)
# 再次闭操作
image_close_two = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, rectKernal)
cv_show('close_two', image_close_two)
# 计算轮廓
thresh_, threshCnts, hierarchy = cv2.findContours(image_close_two.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
# 得到轮廓后在原图展示
cur_img = image
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)
locs = []
# 遍历所有轮廓并筛选
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    print(w)
    # 选合适的区域
    if 2.5 < ar < 4.0:
        # 合适的留下
        if (55 > w > 30) and (20 > h > 10):
            locs.append((x, y, w, h))
# 排序 sorted(iterable可迭代对象, key用来进行比较的元素, reverse=False排序规则默认为False升序)
locs = sorted(locs, key=lambda x: x[0], reverse=False)
output = []
for(i, (gx, gy, gw, gh)) in enumerate(locs):
    groupOutput = []
    # 对轮廓的区域多拿一些，扩大取轮廓
    group = image_gray[gy-8:gy+gh+8, gx-8:gx+gh+8]
    cv_show('group'+str(i), group)
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    thresh_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 58))
        cv_show('roi', roi)
        scores = []
        # 以列表返回可遍历的元祖数组
        for (digit, digitROI) in digits.items():
            # 模板匹配 cv2.matchTemplate(src输入图像, image模板图像, 模板匹配方法)
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # np.argmax()获取array的某一个维度中数值最大的那个元素的索引
        groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(image, (gx-5, gy-5), (gx+gw+5, gy+gh+5), (0, 0, 255), 1)
    # 添加对应的文字 cv2.putText(img图片, word要添加的文字, 添加的位置, 字体, 字体大小, 颜色, 粗细)
    cv2.putText(image, "".join(groupOutput), (gx, gy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    output.append(groupOutput)
# 打印出结果
print("Credit Card Type:{}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
