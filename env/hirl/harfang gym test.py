import os
import yaml
import numpy as np
import time

from environments.HarfangEnv_GYM import HarfangEnv
import environments.dogfight_client as df

def make_env(env_name="test_env"):
    # Ortamı oluşturup döndürür
    env = HarfangEnv()
    return env

def main():
    # Port'u doğrudan buradan ayarla
    port = 50888  # Burayı istediğin porta ayarlayabilirsin

    # local_config.yaml dosyasını oku
    with open('local_config.yaml', 'r') as file:
        local_config = yaml.safe_load(file)
    if local_config['network']['ip'] == 'YOUR_IP_ADDRESS':
        raise ValueError("Please update the 'network.ip' field in local_config.yaml with your own IP address.")

    # Harfang ortamına bağlan
    df.connect(local_config["network"]["ip"], port)
    start = time.time()
    df.disable_log()
    name = "Harfang_GYM"

    # Ortamı başlat (örnek)
    env = make_env()
    #print("Ortam başarıyla oluşturuldu:", env)

    # Basit bir test/reset/run (örnek)
    state = env.reset()
    #print("İlk gözlem:", state)

    # Kısa bir episode simülasyonu (örnek)
    done = False
    step = 0

    while not done and step < 10:
        action = [0, 0, 0, -1]  # Random aksiyon
        next_state, reward, done, info, _ = env.step(action)



        #print(f"Step {step}: reward={reward}, done={done}")
        step += 1
    plane_state = state[20]



if __name__ == '__main__':
    main()
