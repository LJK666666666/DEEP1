import cv2

def main():
    image=cv2.imread("D:\\Qt_code\\smart_video\\python_code\\origin.jpg")
    cv2.imwrite("D:\\Qt_code\\smart_video\\python_code\\result.jpg",image)
    f = open('D:\\Qt_code\\smart_video\\python_code\\flag.txt', 'w')
    f.write("true")
    f.close()

if __name__ == '__main__':
    main()