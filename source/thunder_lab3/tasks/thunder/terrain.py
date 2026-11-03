"""Thunder's eight terrain definitions with Go2's native generator/importer utilities."""
import copy
import math
import numpy as np
import trimesh
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from go2_rl_lab.tasks.mixed.terrain import PerSubTerrainSlopeThresholdGenerator,MeshFlatSlabTerrainCfg
from go2_rl_lab.tasks.mixed.terrain_mesh import MixedTerrainImporter
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


class ThunderTerrainImporter(MixedTerrainImporter):
    def _compute_env_origins_curriculum(self,num_envs,origins):
        result=super()._compute_env_origins_curriculum(num_envs,origins)
        level=self.cfg.fixed_level
        if level is None:return result
        if not 0<=level<origins.shape[0]:raise ValueError('fixed terrain level outside generated rows')
        self.terrain_levels[:]=level
        return origins[self.terrain_levels,self.terrain_types]


@configclass
class ThunderTerrainImporterCfg(TerrainImporterCfg):
    class_type: type=ThunderTerrainImporter
    fixed_level: int|None=None


def pyramid_slope(difficulty, cfg):
    width, length = cfg.size
    if abs(width-length) > 1e-9:
        raise ValueError('Thunder pyramid slope needs a square tile')
    half = cfg.platform_width/2; run = width/2-half
    if run <= 0: raise ValueError('platform must fit inside tile')
    slope = cfg.slope_range[0]+difficulty*(cfg.slope_range[1]-cfg.slope_range[0])
    z = (-1 if cfg.inverted else 1)*run*slope
    lo,hi=width/2-half,width/2+half
    outer=[(0,0),(width,0),(width,length),(0,length)]
    inner=[(lo,lo),(hi,lo),(hi,hi),(lo,hi)]
    vertices=[(x,y,0) for x,y in outer]+[(x,y,z) for x,y in inner]+[(x,y,min(0,z)-.1) for x,y in outer]
    faces=[]
    for i in range(4):
        j=(i+1)%4;faces.extend([(i,j,4+j),(i,4+j,4+i),(i,8+i,8+j),(i,8+j,j)])
    faces.extend([(4,5,6),(4,6,7),(8,10,9),(8,11,10)])
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    if not mesh.is_watertight:raise ValueError('nonclosed slope mesh')
    return [mesh],np.array([width/2,length/2,z])


def progressive_rough(difficulty,cfg):
    from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
    local=copy.deepcopy(cfg);lo,hi=local.noise_range;step=local.vertical_scale
    local.noise_range=(lo,lo+int((hi-lo)/step*difficulty)*step);local.noise_step=step
    return random_uniform_terrain(difficulty,local)


def terrain_cfg():
    return TerrainGeneratorCfg(class_type=PerSubTerrainSlopeThresholdGenerator,
        size=(8.,8.),border_width=20.,border_height=1.,num_rows=10,num_cols=20,
        horizontal_scale=.1,vertical_scale=.005,slope_threshold=.75,
        difficulty_range=(0.,1.),use_cache=False,curriculum=True,color_scheme='none',
        sub_terrains={
            'flat':MeshFlatSlabTerrainCfg(proportion=.10),
            'pyramid_stairs':terrain_gen.MeshPyramidStairsTerrainCfg(proportion=.15,border_width=1.,step_height_range=(.05,.25),step_width=.3,platform_width=3.,holes=False),
            'pyramid_stairs_inv':terrain_gen.MeshInvertedPyramidStairsTerrainCfg(proportion=.15,border_width=1.,step_height_range=(.05,.25),step_width=.3,platform_width=3.,holes=False),
            'boxes':terrain_gen.MeshRandomGridTerrainCfg(proportion=.15,grid_width=.45,grid_height_range=(.05,.20),platform_width=2.,holes=False),
            'random_rough':terrain_gen.HfRandomUniformTerrainCfg(proportion=.15,border_width=.25,noise_range=(.02,.10),noise_step=.02,function=progressive_rough),
            'hf_pyramid_slope':terrain_gen.HfPyramidSlopedTerrainCfg(proportion=.10,border_width=.25,slope_range=(0.,math.tan(math.radians(40))),platform_width=2.,function=pyramid_slope),
            'hf_pyramid_slope_inv':terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=.10,border_width=.25,slope_range=(0.,math.tan(math.radians(40))),platform_width=2.,function=pyramid_slope),
            'wave':terrain_gen.HfWaveTerrainCfg(proportion=.10,border_width=.25,amplitude_range=(.02,.10),num_waves=3),
        })
