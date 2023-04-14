import cv2
import matplotlib.pyplot as plt
import urllib.request

haarcascade_url = 'https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml'
haar_name = "cars.xml"
urllib.request.urlretrieve(haarcascade_url, haar_name)
detector = cv2.CascadeClassifier()


def plt_show(image, title="", gray=False, size=(12, 10)):
    from pylab import rcParams
    temp = image

    # convert to grayscale images
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    # change image size
    rcParams['figure.figsize'] = [7, 7]
    # remove axes ticks
    plt.axis("off")
    plt.title(title)
    plt.imshow(temp, cmap='gray')
    plt.show()


def detect_obj(image):
    # clean your image
    plt_show(image)
    # detect the car in the image
    object_list = detector.detectMultiScale(image)
    print(object_list)
    # for each car, draw a rectangle around it
    for obj in object_list:
        (x, y, w, h) = obj
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # line thickness
    # let's view the image
    plt_show(image)


# we will read in a sample image
image_url = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/Dataset/car-road-behind.jpg"
image_name = "car-road-behind.jpg"
urllib.request.urlretrieve(image_url, image_name)
image = cv2.imread(image_name)

plt_show(image)
detect_obj(image)
