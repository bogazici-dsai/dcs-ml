import time
from hirl.environments.HarfangEnv_GYM_TEST import HarfangEnv
import hirl.environments.dogfight_client as df

def safe_call(func, *args, max_attempts=8, wait=0.25, must_have_key=None):
    for _ in range(max_attempts):
        try:
            result = func(*args)
            if result is not None:
                if must_have_key is None or must_have_key in result:
                    return result
        except Exception:
            pass
        time.sleep(wait)
    return None

def fire_all_slots_and_track(plane_id, label="ALLY", steps=20):
    print(f"\n===== [{label}] TÜM SLOT'LAR ATEŞLENİYOR ve MISSILE TAKİBİ =====")
    slots_pre = safe_call(df.get_machine_missiles_list, plane_id) or []
    missiles_pre = set(safe_call(df.get_missiles_list, must_have_key=None) or [])
    print(f"  [{label}] Slotlar (başlangıç): {slots_pre}")
    print(f"  [{label}] Missile ID'leri (başlangıç): {missiles_pre}")

    if not slots_pre:
        print(f"  [{label}] Slot bulunamadı! Ammo/config kontrol et.")
        return

    for idx, slot in enumerate(slots_pre):
        print(f"\n  🔥 [{label}] Slot {slot} (index {idx}) fırlatılıyor...")
        try:
            df.fire_missile(plane_id, idx)
            df.update_scene()
            time.sleep(1.0)
        except Exception as ex:
            print(f"    [HATA] fire_missile sırasında: {ex}")
            continue

        missiles_now = set(safe_call(df.get_missiles_list, must_have_key=None) or [])
        new_missiles = missiles_now - missiles_pre
        print(f"    ➤ [{label}] Yeni oluşan missile ID'leri: {new_missiles}")

        if not new_missiles:
            print(f"    [{label}] Fırlatmadan sonra yeni missile oluşmadı! (Ammo bitmiş veya simülasyon hatası olabilir)")
            continue

        for mid in new_missiles:
            print(f"    [{label}] Missile ID {mid} pozisyonu takip ediliyor ({steps} adım):")
            for step in range(steps):
                state = safe_call(df.get_missile_state, mid, must_have_key=None)
                if state and "position" in state and state["position"]:
                    pos = state["position"]
                    print(f"      [{label}_MISSILE:{mid}] Step {step:02d}: x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}")
                else:
                    print(f"      [{label}_MISSILE:{mid}] Step {step:02d}: Pozisyon yok")
                time.sleep(0.15)
            print("")
        missiles_pre.update(new_missiles)

def main():
    IP = "10.1.110.30"
    PORT = 50888

    print(f"\n🔗 Dogfight server'a bağlanılıyor... {IP}:{PORT}")
    df.connect(IP, PORT)
    time.sleep(1)

    env = HarfangEnv()
    print("\n✅ HarfangEnv başlatıldı.")

    obs = env.reset()
    print("\n🔄 Reset sonrası gözlem vektörü:")
    print(obs)
    ally_id = env.Plane_ID_ally
    enemy_id = env.Plane_ID_oppo

    fire_all_slots_and_track(ally_id, label="ALLY", steps=20)
    fire_all_slots_and_track(enemy_id, label="ENEMY", steps=20)

    df.disconnect()
    print("\n🛑 Bağlantı kapatıldı.")

if __name__ == "__main__":
    main()
