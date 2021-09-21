import numpy as np
import imageio

env_map = 0.5 * np.ones([512, 1024, 3], dtype=np.float32)
imageio.imwrite('uniform_lighting.hdr', env_map)