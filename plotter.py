import argparse
import itertools
import pprint
import requests
import time
import cv2
import numpy as np


class APIError(Exception):
    """An API Error Exception"""

    def __init__(self, status):
        self.status = status

    def __str__(self):
        return "APIError: {}".format(self.status)


class FirenodejsAPI:
    def __init__(self, url, fake_move=False):
        self._url = url
        self._fake_move = fake_move

        if not fake_move:
            response = self.position(request={'sys':''})
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(response)

            self.position(request={'hom':''})

    
    def camera(self, src='video0'):
        '''Get image from firenodejs REST API.'''
        resp = requests.get(self._url + '/camera/' + src + '/image.jpg')
        if resp.status_code != 200:
            raise APIError('GET /camera/{}/image.jpg {}'.format(src, resp.status_code))
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), -1)


    def position(self, request={}):
        if request:
            if self._fake_move:
                print(request)
                return
            resp = requests.post(self._url + '/firestep', json=request)
            if resp.status_code != 200:
                raise APIError('POST {}/firestep json={} -> {}'.format(self._url, request, resp.status_code))
            time.sleep(0.1)
            return resp.json()


#resp = requests.get('http://10.0.0.10:8080/images/default/image.jpg')
#resp = requests.get('http://10.0.0.10:8080/firestep/model')
#if resp.status_code != 200:
#    raise APIError('GET /firestep/model/ {}'.format(resp.status_code))
#print(resp.content)
#for item in resp.json():
#    print('{}: {}'.format(item, resp.json()[item]))


def set_z(api, z):
    api.position({'mov': {'z': z}})


def move_to_xy(api, x, y):
    api.position({'mov': {'x': int(x), 'y': int(y)}})


def draw_rectangle(api, x0, y0, x1, y1, *, fill=False):
    # outline
    move_to_xy(api, x0, y0)
    set_z(api, -10)
    move_to_xy(api, x0, y1)
    move_to_xy(api, x1, y1)
    move_to_xy(api, x1, y0)
    move_to_xy(api, x0, y0)

    # fill
    if fill:
        dx = 1
        move_to_xy(api, min(x0, x1), y0)
        for x in range(min(x0, x1), max(x0, x1)+dx, dx):
            move_to_xy(api, x, y0)
            move_to_xy(api, x, y1)
    set_z(api, 0)


def draw_chessboard(api, args):
    chessboard(api, (0, 0), args.size)


def chessboard(center, size):
    '''Draws a 8x8 chessboard.'''
    a = size / 8
    x0 = int(floor(center[0] - size / 2))
    y0 = int(floor(center[1] - size / 2))
    x1 = int(ceil(center[0] + size / 2))
    y1 = int(ceil(center[1] + size / 2))

    draw_rectangle(x0, y0, x1, y1)
    for x, y in itertools.product(range(-4, 4), range(-4, 4)):
        if (x + y) % 2 != 0:
            continue
        draw_rectangle(x * a, y * a, (x + 1) * a, (y + 1) * a, fill=True)


def draw_bitmap(api, args):
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    bitmap(api, img, (0, 0), scale=1.0)


def bitmap(api, img, center, *, threshold=128, scale=1, h_flip=True):
    '''
    Draws binary image using dots where a pixel value is less than threshold
    1px = 1mm^2
    '''
    assert len(img.shape) == 2
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    h, w = img.shape
    x0, y0 = center - np.asarray((int(h / 2), int(w / 2)))
    print('Drawing bitmap {}x{} at {}:{}'.format(w, h, x0, y0))
    if h_flip:
        img = img[:, ::-1]
    for r in range(h):
        set_z(api, 0)
        for c in range(w):
            if img[r, c] < threshold:
                api.position({'mov': {'x': int(c + x0), 'y': int(r + y0)}})
                set_z(api, -10)
                set_z(api, -5)
        set_z(api, 0)


def draw_bitmap2(api, args):
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    dots = bitmap_2_dots(img)  # threshold
    draw_dots(api, dots)  # z_up, z_down


def bitmap_2_dots(img, *, threshold=128):
    """Convert bitmap image into a generator of point coordinates."""
    assert len(img.shape) == 2
    h, w = img.shape
    for r, c in itertools.product(range(h), range(w)):
        if img[r, c] < threshold:
            yield (c, r)


def draw_dots(api, dots, *, z_up=0, z_down=-10):
    for x, y in dots:
        api.position({'mov': {'x': x, 'y': y}})
        set_z(api, z_down)
        set_z(api, z_up)


def bitmap_lines(img, center, *, threshold=128, scale=1, h_flip=True):
    '''
    Draws binary image using horizontal lines where a line is present if pixel value is less than threshold
    1px = 1mm^2
    '''
    assert len(img.shape) == 2
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    h, w = img.shape
    x0, y0 = center - np.asarray((int(h / 2), int(w / 2)))
    print('Drawing bitmap {}x{} at {}:{}'.format(w, h, x0, y0))
    if h_flip:
        img = img[:, ::-1]
    for r in range(h):
        row = img[r, :]
        one_indices = numpy.where(row >= threshold)
        zero_indices = numpy.where(row < threshold)
        if zero_indices[0] > one_indices[0]:
            one_indices = one_indices[1:]
        
        set_z(0)
        for (begin, end) in zip(zero_indices, one_indices):
            print(begin, end)
            api.position({'mov': {'x': int(begin + x0), 'y': int(r)}})
            set_z(-10)
            api.position({'mov': {'x': int(end + x0), 'y': int(r)}})
            set_z(-5)
        set_z(0)


parser = argparse.ArgumentParser(description='Tool for drawing via firenodejs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers()
parser.add_argument('--url', default='http://10.0.0.10:8080', help='URL of firenodejs')
parser.add_argument('--fakemove', action='store_true', help='Fake movement')

parser_bitmap = subparsers.add_parser('bitmap', help='Bitmap drawing')
parser_bitmap.add_argument('--image', help='Path to image', required=True)
parser_bitmap.set_defaults(func=draw_bitmap)

parser_chess = subparsers.add_parser('chess', help='Chessboard drawing')
parser_chess.add_argument('--size', help='Width of the chessboard', default=80)
parser_chess.set_defaults(func=draw_chessboard)


args = parser.parse_args()

api = FirenodejsAPI(args.url, args.fakemove)

args.func(api, args)



#feed = 'video0'
#print('Press ESC to end')
#while True:
#    image = api.camera(src=feed)
#    cv2.imshow(feed, image)
#    k = cv2.waitKey(1)
#    if k == 27:
#        break
