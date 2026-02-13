import cv2
import numpy as np
import time
from collections import deque
import tensorflow as tf

# ---------- Настройки ----------
MODEL_PATH = "best_model.keras"   # путь к модели
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
INPUT_SIZE = (48, 48)            # размер для модели
COLOR_MODE = "grayscale"         # модель на grayscale
SMOOTH_QUEUE = 7                 # сколько последних предсказаний учитывать
CONFIDENCE_THRESHOLD = 0.35      # минимальная уверенность для показа метки
SCALE_FACTOR = 1.3               # масштаб для детектора лиц (увеличивает область поиска)
MIN_NEIGHBORS = 5
# -------------------------------

# Метки (порядок должен совпадать с обученной моделью)
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Загружаем модель
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# Загружаем каскад
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("Не удалось загрузить Haar cascade. Проверь CASCADE_PATH: " + CASCADE_PATH)

# вспомогательные структуры для сглаживания прогнозов по каждому лицу
# будем хранить для каждого обнаруженного лица очередь последних softmax-вероятностей
# простая реализация: привязка по позициям прямоугольников
smoothing = []  # list of dicts: {'bbox':(x,y,w,h), 'probs_deque':deque([...])}

# Функция предобработки ROI для модели
def preprocess_face(face_img):
    # face_img: grayscale or color BGR region
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        # конвертим в grayscale (т.к. модель была на grayscale)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    face_resized = face_resized.astype('float32') / 255.0
    # модель ожидает форму (1,48,48,1)
    face_resized = np.expand_dims(face_resized, axis=-1)
    face_resized = np.expand_dims(face_resized, axis=0)
    return face_resized

# Функция для привязки/обновления очередей сглаживания
def match_and_update_smoothing(bboxes, probs_vectors):
    # bboxes: list of (x,y,w,h) текущих обнаружений
    # probs_vectors: list of np.array (softmax vectors) размер 7
    global smoothing
    new_smoothing = []

    used = [False]*len(bboxes)

    # match existing smoothing entries to new detections via IoU-ish (по центрам)
    for s in smoothing:
        sx, sy, sw, sh = s['bbox']
        scx, scy = sx + sw/2, sy + sh/2
        best_idx = -1
        best_dist = 1e9
        for i, (x,y,w,h) in enumerate(bboxes):
            if used[i]: continue
            cx, cy = x + w/2, y + h/2
            dist = (cx-scx)**2 + (cy-scy)**2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx != -1 and best_dist < (max(sw, sh)*max(sw, sh))*4: # threshold
            # обновляем
            used[best_idx] = True
            probs = s['probs_deque'].copy()
            probs.append(probs_vectors[best_idx])
            if len(probs) > SMOOTH_QUEUE:
                probs.popleft()
            new_smoothing.append({'bbox': bboxes[best_idx], 'probs_deque': probs})
        # иначе — если не нашли подходящее соответствие — не сохраняем старый (лицо ушло)

    # добавляем новые, не сопоставленные детекции
    for i, b in enumerate(bboxes):
        if not used[i]:
            dq = deque(maxlen=SMOOTH_QUEUE)
            dq.append(probs_vectors[i])
            new_smoothing.append({'bbox': b, 'probs_deque': dq})

    smoothing = new_smoothing

def list_cameras(max_devices=10):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

cams = list_cameras()
if not cams:
    raise RuntimeError("Камеры не найдены")

print("Доступные камеры:")
for i in cams:
    print(f"[{i}] Камера {i}")

cam_id = int(input("Выберите номер камеры: "))

# Запуск видеопотока
cap = cv2.VideoCapture(cam_id)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

prev_time = time.time()
fps = 0

print("Запуск детектора. Нажми 'q' чтобы выйти.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detectMultiScale — простой, но быстрый метод
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(30,30)
    )

    probs_vectors = []
    bboxes = []
    # проходим по лицам, предсказываем
    for (x,y,w,h) in faces:
        # optional: расширить bbox немного, чтобы захватить больше контекста
        pad_w = int(0.1*w)
        pad_h = int(0.1*h)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        face_roi = frame[y1:y2, x1:x2]
        inp = preprocess_face(face_roi)
        preds = model.predict(inp, verbose=0)  # shape (1,7)
        probs = preds[0]
        probs_vectors.append(probs)
        bboxes.append((x1, y1, x2 - x1, y2 - y1))

    # обновляем структуру сглаживания и получаем итоговые прогнозы
    if len(bboxes) > 0:
        match_and_update_smoothing(bboxes, probs_vectors)

    # рисуем результаты
    for s in smoothing:
        x,y,w,h = s['bbox']
        # усредняем вероятности по очереди
        probs_stack = np.stack(s['probs_deque'], axis=0)  # (k,7)
        avg_probs = probs_stack.mean(axis=0)
        top_idx = int(np.argmax(avg_probs))
        conf = float(avg_probs[top_idx])

        # только если уверенность выше порога — показываем метку
        label_text = ""
        if conf >= CONFIDENCE_THRESHOLD:
            label_text = f"{EMOTION_LABELS[top_idx]} {conf*100:.0f}%"
        else:
            label_text = f"Unknown {conf*100:.0f}%"

        # draw rectangle and label
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        # background for text
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y-25), (x+tw+6, y), (0,255,0), -1)
        cv2.putText(frame, label_text, (x+3, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    cv2.putText(frame, "Q - exit | C - change camera",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS
    cur_time = time.time()
    fps = 0.9*fps + 0.1*(1.0/(cur_time - prev_time)) if prev_time else 0
    prev_time = cur_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Emotion recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # переключение камеры
        cam_id = (cam_id + 1) % len(cams)
        cap.release()
        cap = cv2.VideoCapture(cams[cam_id])
        print(f"Переключено на камеру {cams[cam_id]}")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
