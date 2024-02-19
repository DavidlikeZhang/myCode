import cv2
import matplotlib.pyplot as plt
import numpy as np


class Cv2Test:
    """"一次模拟cv2实际应用的尝试"""

    def __init__(self):
        self.desk_ = cv2.imread('imgs/desk.jpg')
        self.desk = cv2.resize(self.desk_, (1200, 800))
        # self.rou = cv2.imread('D:/opencv_Files/imagine/round.png')
        # self.rou_seed = cv2.imread('D:/opencv_Files/imagine/round_seed.png')
        self.book = cv2.imread('imgs/book.jpg')
        self.books = cv2.imread('imgs/books.jpg')

        self.kernel = np.ones((3, 3), np.uint8)  # 定义卷积

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        # waiting time in millisecond,press any key to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def harris_corner_detect(self):
        # 角点检测(运用数学方法在内部、边界、、角点中识别角点)

        img = cv2.imread('imgs/desk.jpg')
        img = cv2.resize(img, (1200, 800))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # Here 0.01 decides how many corners can be detected
        extent = 0.01  # 改变extent的值可以增减角点检测数量
        img[dst > extent*dst.max()] = [0, 0, 255]
        self.cv_show('Harris corner detection', img)

    def canny_figure_detect(self):
        # Canny轮廓检测算法（自带非极大值抑制）

        # Gaussian filter，图像降噪
        aussian = cv2.GaussianBlur(self.desk, (3, 3), 0)
        # 双阈值检测，Cannedge detecton， 改变minVal , maxVal 可以增减细节
        canny = cv2.Canny(aussian, 50, 150)
        self.cv_show('canny', canny)

    def clahe_equalize(self):
        # 图像自适应均衡化(强化亮暗对比)，并展示直方图

        img2 = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
        plt.hist(img2.ravel(), 256)
        plt.show()

        # equalization
        equ = cv2.equalizeHist(img2)
        plt.hist(equ.ravel(), 256)
        plt.show()

        # self-suited equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        res_clahe = clahe.apply(img2)
        plt.hist(equ.ravel(), 256)
        plt.show()

        res = np.hstack((img2, equ, res_clahe))
        self.cv_show('equ', res)

    def open_close(self):
        # 开运算：先腐蚀，后膨胀
        opening = cv2.morphologyEx(self.rou_seed, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # 闭运算：先膨胀，后腐蚀
        closing = cv2.morphologyEx(self.rou_seed, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        titles = ['rou_seed', 'opening', 'closing']
        images = [self.rou_seed, opening, closing]
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(titles[i])
            plt.imshow(images[i], 'gray')

    def brute_force_match(self):
        # Brute Force 暴力匹配: 将特征点进行匹配，将好的匹配连线
        # 一对一的匹配
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.book, None)
        kp2, des2 = sift.detectAndCompute(self.books, None)
        bf = cv2.BFMatcher(crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        res1 = cv2.drawMatches(self.book, kp1, self.books, kp2, matches[:10], None, flags=2)
        self.cv_show('Brute Force 暴力匹配 1 to 1', res1)

        # k对最佳匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        resk = cv2.drawMatchesKnn(self.book, kp1, self.books, kp2, good, None, flags=2)
        self.cv_show('Brute Force 暴力匹配 k', resk)

    def flann_based_matcher(self):
        # Flann Based matcher: 处理大量数据更有优势，将特征点进行匹配，将好的匹配连线
        queryImage = self.book
        trainingImage = self.books

        # 创建sift检测器
        # 创建实例sift（4.0以下版本使用cv2.features2d.SIFT_create()）
        sift = cv2.SIFT_create()
        # kp:key points ,descriptor描述符
        kp1, des1 = sift.detectAndCompute(queryImage, None)
        kp2, des2 = sift.detectAndCompute(trainingImage, None)
        # kp 中包含的信息：
        #   pt：关键点的坐标
        #   angle：角度，表示关键点的方向。即对每个关键点周围的区域计算所得的特征向量
        #   size：该点直径的大小

        # 设置Flannel参数
        FLANN_INDEX_KDTREE = 0
        # indexParams = {'algorithm': 0, 'trees': 5}  指定索引中的树应递归遍历的次数（遍历次数越多，精度越高，时间越长）
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # searchParams = {'checks': 50}
        searchParams = dict(checks=50)
        # 传入两个参数indexParams，searchParams后创建实例flann
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        # matches是一种DMatch结构，含有：
        #   DMatch.distance：描述符之间的距离，代表特征点间的距离，越低越好。
        #   DMatch.queryIdx：主动匹配的描述符组中描述符的索引（即第几个特征点描述符）
        #   DMatch.trainIdx：被匹配的描述符组中描述符的索引（即第几个特征点描述符）
        #   DMatch.imgIdx：目标图像的索引。
        matches = flann.knnMatch(des1, des2, k=2)

        # 准备一个空的掩膜matchesMask来绘制好的匹配
        matchesMask = [[0, 0] for i in range(len(matches))]
        # 向掩膜matchesMask中添加好的匹配
        for i, (m, n) in enumerate(matches):
            # 比值测试：首先获取与 A 距离最近的点 B（最近）和 C（次近），只有当 B/C
            # 小于0.5时，该匹配才被认为是好的匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为 0
            if m.distance / n.distance < 0.5:
                matchesMask[i] = [1, 0]

        # 给特征点和匹配的线定义颜色(r,g,b)
        drawParams = dict(matchColor=(0, 0, 255), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
        # 画出匹配的结果，cv2.drawMatchesKnn 只绘制掩膜matchesMask中标记为好的匹配（matchesMask[i]为[1, 0]）
        resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)
        self.cv_show('Matching Result', resultImage)

    def threshold(self):
        # 阈值处理
        img = cv2.imread('imgs/flower.png')
        # read the picture as grayscale
        img_gray = cv2.imread('imgs/flower.png', cv2.IMREAD_GRAYSCALE)

        # return maxval if beyond thresh,otherwise return 0
        ret1, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        # INV:invert
        ret2, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        # return thresh if beyond thresh,otherwise remain unchanged
        ret3, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
        # return 0 if below thresh,otherwise remain unchanged
        ret4, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
        ret5, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

        # Otsu algorithm thresholding to automatically find the optimal threshold value
        ret, thresh6 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, thresh7 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 三角算法阈值处理
        ret, thresh8 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        ret, thresh9 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

        titles = ['Original image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])  # 可以使坐标数字消失，或调整坐标数字
        plt.show()

        titles2 = ['Original image', 'BINARY', 'THRESH_BINARY \n+ THRESH_OTSU', 'THRESH_BINARY_INV \n+ THRESH_OTSU',
                   'THRESH_BINARY \n+ THRESH_TRIANGLE', 'THRESH_BINARY_INV \n+ THRESH_TRIANGLE']
        images2 = [img_gray, thresh1, thresh6, thresh7, thresh8, thresh9]

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images2[i], 'gray')
            plt.title(titles2[i])
        #    plt.xticks([]),plt.yticks([]) # 可以使坐标数字消失，或调整坐标数字
        plt.show()

    def bounding_rect(self):
        # draw the bounding rect of each figure
        images = cv2.imread('imgs/images.jpg')
        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # opencv4.0 以下版本cv2.findContours() 有三个返回值。opencv4.0 以上两个
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # show the rect one by one
        for i in range(0, len(contours), 2):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            extent = float(cv2.contourArea(cnt) / (w * h))
            img = cv2.rectangle(images.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.cv_show(str(extent), img)


