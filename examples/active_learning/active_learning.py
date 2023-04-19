import time

from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
from waffle_menu.active_learning import PL2NSampling


hub = UltralyticsHub.load(
    name="digit_detector"
)

al_callback = PL2NSampling(
    hub=hub,
    diversity_sampling=True,
    device="0"
).sample(
    image_dir = "images",
    num_images = 10,
    result_dir = "result_dir",
    save_images = False,
    hold = False
)

while not al_callback.is_finished():
    time.sleep(1)
    print(al_callback.get_progress())

print(al_callback.total_file)
