import requests
import cv2
import numpy as np


class FirenodejsAPI:
    def __init__(self, url):
        self._url = url

    
    def get_image(self, src='video0'):
        '''Get image from firenodejs REST API.'''
        resp = requests.get(self._url + '/camera/' + src + '/image.jpg')
        if resp.status_code != 200:
            raise ApiError('GET /camera/{}/image.jpg {}'.format(src, resp.status_code))
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), -1)


#resp = requests.get('http://10.0.0.10:8080/firestep/model')
#resp = requests.get('http://10.0.0.10:8080/images/default/image.jpg')
#if resp.status_code != 200:
#    raise ApiError('GET /firestep/model/ {}'.format(resp.status_code))
#print(resp.content)
#for item in resp.json():
#    print('{}: {}'.format(item, resp.json()[item]))


feed = 'video0'
print('Press ESC to end')
api = FirenodejsAPI('http://10.0.0.10:8080')
while True:
    image = api.get_image(src=feed)
    cv2.imshow(feed, image)
    k = cv2.waitKey(1)
    if k == 27:
        break
