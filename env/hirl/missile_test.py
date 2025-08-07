import hirl.environments.dogfight_client as df
import time
import json

# --------- AYARLAR ---------
IP = "10.1.110.30"
PORT = 50888
MISSILE_SLOT = 0
LIFE_DURATION = 10  # saniye
FRAME_TIME = 1 / 60
RETRY_LIMIT = 10
WAIT_AFTER_FIRE = 0.5  # saniye

# --------- BAĞLANTI ---------
df.connect(IP, PORT)
time.sleep(1)

planes = df.get_planes_list()
if not planes:
    raise RuntimeError("Hiç uçak bulunamadı!")
plane_id = planes[0]
print(f"[INFO] Kontrol edilecek uçak ID: {plane_id}")

df.reset_machine(plane_id)
missiles = df.get_machine_missiles_list(plane_id)
if not missiles:
    raise RuntimeError("Uçakta füze yok!")

missile_id = missiles[MISSILE_SLOT]
print(f"[INFO] Füze ID (slot {MISSILE_SLOT}): {missile_id}")

# --------- MİNİMAL FÜZEYİ TEST ---------
print(f"[DEBUG] Testing basic missile state access:")

# Test 1: Can we access the unfired missile at all?
try:
    state = df.get_missile_state(missile_id)
    print(f"[DEBUG] Unfired missile state accessible: {state is not None}")
except Exception as e:
    print(f"[DEBUG] Unfired missile state failed: {type(e).__name__}: {e}")

# Test 2: Fire missile WITHOUT any other commands
print(f"[DEBUG] Firing missile - no other commands")
try:
    df.fire_missile(plane_id, MISSILE_SLOT)
    print(f"[DEBUG] Fire command sent successfully")
except Exception as e:
    print(f"[ERROR] Fire command failed: {e}")
    df.disconnect()
    exit()

# Test 3: Try to access fired missile state immediately (no update_scene)
try:
    state = df.get_missile_state(missile_id)
    print(f"[SUCCESS] Fired missile state: {state}")
except Exception as e:
    print(f"[DEBUG] Immediate post-fire access failed: {type(e).__name__}: {e}")

print(f"[INFO] Basic test complete")
df.disconnect()