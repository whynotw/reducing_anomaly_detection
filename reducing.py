import numpy as np
import cv2
import time

class ReducingImageProcess():

    def __init__(self,
                 image_origin,
                 size_filter,
                 stride,
                 number_k):
        self.time_create = time.time()
        if size_filter%2 == 0:
            print("size of filter %d should be odd number"%size_filter)
        self.size_filter = size_filter
        self.stride = stride
        self.number_k = number_k

        self.image_origin = image_origin
        height_image, width_image = image_origin.shape[:2]

        remainder = (height_image-self.size_filter)%self.stride
        self.image_origin = self.image_origin[:height_image-remainder,:,:]
        remainder = (width_image-self.size_filter)%self.stride
        self.image_origin = self.image_origin[:,:width_image-remainder,:]
        del remainder

        self.height_image, self.width_image = self.image_origin.shape[:2]
        self.height_reduced = (self.height_image-self.size_filter)/self.stride+1
        self.width_reduced  = (self.width_image -self.size_filter)/self.stride+1
        self.dim_data = 3*self.size_filter**2
        self.image_accumulation = np.zeros((self.height_image,self.width_image,3),
                                           dtype=np.float32)
        self.pixel_count = np.zeros((self.height_image,self.width_image),
                                    dtype=np.uint16)

        tmp = np.arange(0,self.width_reduced*self.stride,self.stride,dtype=np.uint16)
        self.x0_data = np.stack((tmp,)*self.height_reduced)
        self.x1_data = self.x0_data+self.size_filter
        tmp = np.arange(0,self.height_reduced*self.stride,self.stride,dtype=np.uint16)
        self.y0_data = np.stack((tmp,)*self.width_reduced).T
        self.y1_data = self.y0_data+self.size_filter
        del tmp

        self.knn = cv2.ml.KNearest_create()

    def __call__(self):

        self._get_super_pixel()
        self._gather_patches()
        self.image_residual = np.uint8(np.sum(np.abs(np.int16(self.image_origin)
                                                    -np.int16(self.image_reducing)),axis=-1)/3)
        #self.image_residual_blurred = self.image_residual.copy()
        #self.image_residual_blurred = cv2.GaussianBlur(self.image_residual_blurred,(self.size_filter*2+1,)*2,0)
        #mean, sd = cv2.meanStdDev(self.image_residual_blurred)
        #image_normalized = np.abs(self.image_residual_blurred-mean)/sd
        #self.heatmap = np.zeros_like(image_normalized,dtype=np.uint8)
        #for s in range(8):
        #    self.heatmap += np.uint8( (image_normalized>s) *s*8 )
        #self.heatmap = np.uint8((image_normalized>3)*255)
        #ret,self.heatmap = cv2.threshold(self.image_residual_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #self.heatmap = cv2.adaptiveThreshold(self.image_residual_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        self.time_process = time.time()-self.time_create

    def _get_super_pixel(self):
        self.data_total  = np.empty((self.height_reduced,
                                     self.width_reduced,
                                     self.dim_data),
                                     np.uint8)
        self.label_total = np.empty((self.height_reduced,
                                     self.width_reduced,
                                     1),
                                     np.float32)
        count = 0
        for j in range(0,self.height_reduced):
            for i in range(0,self.width_reduced):
                x0 = i*self.stride
                y0 = j*self.stride
                x1 = x0+self.size_filter
                y1 = y0+self.size_filter
                self.data_total[j,i,:] = self.image_origin[y0:y1,x0:x1,:].reshape(-1)
                self.label_total[j,i,0] = count
                count += 1

    #def _get_knn_result_for_single_pixel(self,j,i):
    #    self.knn.train(self.data_train,cv2.ml.ROW_SAMPLE,self.label_train)
    #    local = self.data_total[j,i,:].reshape(1,-1).astype(np.float32)
    #    ret, results, neighbours, distances = self.knn.findNearest(local,self.number_k)
    #    neighbours = neighbours[0].astype(np.int64)
    #    distances = distances[0]
    #    distances -= distances[0]
    #    return neighbours, distances

    def _get_knn_results(self):
        tmp = self.data_total.reshape((-1,self.dim_data)).astype(np.float32)
        self.knn.train(tmp,cv2.ml.ROW_SAMPLE,self.label_total.reshape(-1,1))
        tmp0 = tmp[0]
        time0 = time.time()
        ret, results, neighbours, distances = self.knn.findNearest(tmp0.reshape(1,self.dim_data),1)
        time_try = time.time()-time0
        print(time_try*tmp.shape[0])
        ret, results, neighbours, distances = self.knn.findNearest(tmp,self.size_filter**2+self.number_k)
        del tmp
        neighbours = neighbours.astype(np.int64)
        return neighbours, distances

    #def _ignore_myself(self,j,i):
    #    #self.data_train = self.data_total.copy().reshape((-1,self.dim_data)).astype(np.float32)
    #    #self.label_train = self.label_total.reshape((-1,1))

    #    x0 = self.stride*i
    #    y0 = self.stride*j
    #    x1 = x0+self.size_filter
    #    y1 = y0+self.size_filter

    #    mask = (self.x0_data>x1)|(self.x1_data<x0)|(self.y0_data>y1)|(self.y1_data<y0)
    #    self.label_train = self.label_total[mask].reshape((-1,1))
    #    mask = np.stack((mask,)*self.dim_data,axis=-1)
    #    self.data_train = self.data_total[mask].reshape((-1,self.dim_data)).astype(np.float32)

    def _gather_patches(self):
        count = 0
        neighbours,distances = self._get_knn_results()
        for j in range(0,self.height_reduced):
            for i in range(0,self.width_reduced):
                patch = np.zeros((self.size_filter,self.size_filter,3),dtype=np.float32)
                x0 = self.stride*i
                y0 = self.stride*j
                x1 = x0+self.size_filter
                y1 = y0+self.size_filter
                partition = 1e-7
                effective = -1
                for k in range(len(neighbours)):
                    #print(i,j,k)
                    neighbour = neighbours[count][k]
                    j_neighbour = int(neighbour/self.width_reduced)
                    i_neighbour =     neighbour%self.width_reduced
                    distance = distances[count][k]
                    x0_neighbour = self.stride*i_neighbour
                    y0_neighbour = self.stride*j_neighbour
                    x1_neighbour = x0_neighbour+self.size_filter
                    y1_neighbour = y0_neighbour+self.size_filter
                    #print(x0,x1,y0,y1)
                    #print(x0_neighbour,x1_neighbour,y0_neighbour,y1_neighbour)
                    if x0>x1_neighbour or x1<x0_neighbour or y0>y1_neighbour or y1<y0_neighbour:
                        effective += 1
                    else:
                        continue
                    if effective == 0:
                        distance0 = distance
                    if effective >= self.number_k:
                        break
                    distance -= distance0
                    factor = np.exp(-distance/self.dim_data)
                    if factor > 2**-8:
                        #print(factor)
                        patch += factor * self.data_total[j_neighbour,i_neighbour,:].reshape(
                                          (self.size_filter,self.size_filter,3))
                        partition += factor
                self.image_accumulation[y0:y1,x0:x1,:] += patch/partition
                self.pixel_count[y0:y1,x0:x1] += 1
                count += 1
        self.image_reducing = np.uint8(self.image_accumulation/np.stack((self.pixel_count,)*3,axis=-1))

if __name__ == "__main__":

    #imagename = "red_in_green02.jpg"
    #imagename = "/data/dataset/20190812_PCB_defect/AIO/9.jpg"
    imagename = "/data/dataset/20190812_PCB_defect/CAD/CROP_LT_30.745800,31.951400_RB_32.170800,33.851400-defIdx-1_pasteBox_(274, 424, 285, 430)-0717_163619.438709.png"
    image_origin = cv2.imread(imagename)
    height, width = image_origin.shape[:2]
    scale_reduction = 0.5
    image_origin = cv2.resize(image_origin,(int( width*scale_reduction),
                                            int(height*scale_reduction)))
    cv2.imshow("image0",image_origin)
    key = cv2.waitKey(1)
    time0 = time.time()
    reducing = ReducingImageProcess(image_origin = image_origin,
                                          size_filter = 7,
                                          stride = 3,
                                          number_k = 5)
    print("size of original image: %d x %d"%(reducing.height_image  ,reducing.width_image  ))
    print("size of reduced image : %d x %d"%(reducing.height_reduced,reducing.width_reduced))

    reducing()
    print("process time: %f s"%(reducing.time_process))

    cv2.imshow("image1",reducing.image_reducing)
    cv2.imshow("residual",reducing.image_residual)
    #cv2.imshow("heatmap",reducing.heatmap)
    key = cv2.waitKey(0)
