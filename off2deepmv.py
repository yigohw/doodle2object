import os, sys
import open3d as o3d
import numpy as np
import torch
from PIL import Image

model = sys.argv[1]
mesh = o3d.io.read_triangle_mesh(model)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
ctr = vis.get_view_control()
mv = []
for j in range(12):
    ctr.rotate(30*j, 0)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    
    depth = vis.capture_depth_float_buffer()

    img = Image.fromarray(np.asarray(depth))
    box = (430, 0, 430+1055, 1055)
    img=img.convert('L').crop(box).resize((128, 128))
    mv.append(np.array(img))

vis.destroy_window()
mv = torch.tensor(np.array(mv)).to(torch.float32)
mv = (mv-mv.min())/(mv.max()-mv.min())
print(mv.shape)
torch.save(mv, sys.argv[2])
