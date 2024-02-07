# -*- coding: utf-8 -*-
# @File : CA.py
# @desc : attention please, all the simulation results are based on the previous two years data 请注意，result文件夹中的模拟结果均基于前两年的土地利用数据

import shapely
import geopandas as gpd
import os
import glob
from tqdm import tqdm
from geopy.distance import geodesic
import math
import osmnx as ox
from shapely.geometry import Point, LineString, Polygon
import random
import numpy as np
from shapely.ops import nearest_points
import networkx as nx
# ['英文','中文',土地资源价值指数潜在价值F，土地利用转化难度指数土地更新周期T]
newID_DLMC = {0: ['Commercial and Service Facilities', '商业服务业设施用地',0.1,3],
              1: ['Green Space', '绿化用地',0.1,3],
              2: ['Land for industrial parks', '产业园区用地',0.1,3],
              3: ['Land to be developed', '待开发用地',0.5,0.3],
              4: ['Public Administration', '公共管理与公共服务设施用地',0.1,3],
              5: ['Residential land', '居住用地',0.1,3],
              6: ['Rural land', '农村用地',0.1,3]}


def calculate_land_diffuse_attenuation_index(shp_file,station_point):
    '''
    计算土地扩散衰减指数
    :param shp_file:
    :param station_point:
    :return:
    '''
    ldai = []
    for i in tqdm(range(shp_file.shape[0])):
        parcel = shp_file.iloc[i]
        # 计算地块中心点与曹庄地铁站距离
        parcel_centroid = parcel['geometry'].centroid
        data_cen_caozhuang = gpd.GeoDataFrame({'geometry':[parcel_centroid,station_point]},crs='epsg:4326').to_crs('epsg:5234')
        distance = round(data_cen_caozhuang.distance(data_cen_caozhuang.shift())[1]/1000,3)
        # 计算扩散衰减指数
        a = 0.5
        l = 0
        land_diffuse_attenuation_index = math.pow(math.e,-1*a*(distance+l))
        ldai.append(land_diffuse_attenuation_index)
    shp_file['LDAI'] = ldai
    return shp_file

def calculate_land_resource_value_index(land_file,road_network,poi):
    '''
    计算土地资源价值指数
    :param land_file:
    :param road_network:
    :param poi:
    :return:
    '''
    lrvi = []
    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        road_intersection = gpd.overlay(road_network,gpd.GeoDataFrame(geometry=[parcel.geometry],crs='epsg:4326'),how='intersection')
        road_length_sum = round(road_intersection['length'].sum()/1000,4)
        poi_intersection = gpd.overlay(poi,gpd.GeoDataFrame(geometry=[parcel.geometry],crs='epsg:4326'),how='intersection')
        poi_num = poi_intersection.shape[0]
        parcel_area = round(gpd.GeoDataFrame(geometry=[parcel.geometry],crs='epsg:4326').to_crs('epsg:5234').area[0]/1e6,4)
        parcel_land_use = parcel['newID']
        parcel_F = newID_DLMC[parcel_land_use][2]
        a1,a2,a3 = 0.5,0.3,0.2
        land_resource_value_index = a1*(road_length_sum/parcel_area) + a2*(poi_num/parcel_area) + a3*(parcel_F)
        lrvi.append(land_resource_value_index)
    land_file['LRVI'] = lrvi
    return land_file

def calculate_land_use_change_index(land_file):
    '''
    计算土地利用转变指数
    :param land_file:
    :return:
    '''
    luci = []
    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        parcel_area = round(gpd.GeoDataFrame(geometry=[parcel.geometry],crs='epsg:4326').to_crs('epsg:5234').area[0]/1e6,4)
        parcel_land_use = parcel['newID']
        parcel_T = newID_DLMC[parcel_land_use][3]
        if parcel_land_use != 3: parcel_s_in = random.uniform(0.5,1.0)*parcel_area
        else: parcel_s_in = 0.05*parcel_area
        land_use_change_index = (1/1+ math.pow(math.e,parcel_T))*(parcel_s_in/parcel_area)
        luci.append(land_use_change_index)
    land_file['LUCI'] = luci
    return land_file

def nomalize_indexs(land_file):
    '''
    归一化指数并计算最终潜力值
    :param land_file:
    :return:
    '''
    land_file['LDAI_NOR'] = (land_file['LDAI']-land_file['LDAI'].min())/(land_file['LDAI'].max()-land_file['LDAI'].min())
    land_file['LRVI_NOR'] = (land_file['LRVI'] - land_file['LRVI'].min()) / (
                land_file['LRVI'].max() - land_file['LRVI'].min())
    land_file['LUCI_NOR'] = (land_file['LUCI'] - land_file['LUCI'].min()) / (
            land_file['LUCI'].max() - land_file['LUCI'].min())

    potential_final = []

    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        parcel_potential = parcel['LDAI_NOR'] + parcel['LRVI_NOR'] + parcel['LUCI_NOR']
        potential_final.append(parcel_potential)
    land_file['POTENTIAL_FINAL'] = potential_final
    land_file['POTENTIAL_FINAL_NOR'] = (land_file['POTENTIAL_FINAL'] - land_file['POTENTIAL_FINAL'].min()) / (
            land_file['POTENTIAL_FINAL'].max() - land_file['POTENTIAL_FINAL'].min())
    return land_file

def calculate_spatial_attraction(land_file,current_time,predict_time):
    '''
    空间吸引力
    :param land_file:
    :param current_time:
    :param predict_time:
    :return:
    '''
    spatial_attraction = []
    parcel_num = land_file.shape[0]
    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        # 计算地块中心点与曹庄地铁站距离
        parcel_centroid = parcel['geometry'].centroid
        parcel_distances = []
        for j in range(land_file.shape[0]):
            if j==i: continue
            other_parcel = land_file.iloc[j]
            other_parcel_centroid = other_parcel['geometry'].centroid
            data_p_o_centroid = gpd.GeoDataFrame({'geometry':[parcel_centroid,other_parcel_centroid]},crs='epsg:4326').to_crs('epsg:5234')
            distance = round(data_p_o_centroid.distance(data_p_o_centroid.shift())[1] / 1000, 4)
            parcel_distances.append(distance)
        parcel_distances = sum(parcel_distances)
        if parcel_num >2:
            sa = ((2*((parcel_distances/(parcel_num-1))-1))/(parcel_num-2))/(predict_time-current_time)
        else:sa = 0
        spatial_attraction.append(sa)
    land_file['SA'] = spatial_attraction
    return land_file

def calculate_develop_attraction(land_file,poi,current_time,predict_time):
    '''
    发展吸引力
    :param land_file:
    :param poi:
    :param current_time:
    :param predict_time:
    :return:
    '''
    attraction_weight = {'美食':0.1,'休闲娱乐':0.2,'酒店':0.1,'生活设施':0.2,'医疗':0.5,'公司':0.2,'体育文化':0.1,'教育':0.5}
    develop_attraction = []

    type_num = []
    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        poi_intersection = gpd.overlay(poi, gpd.GeoDataFrame(geometry=[parcel.geometry], crs='epsg:4326'),
                                       how='intersection')
        poi_type_num = poi_intersection['type'].unique().shape[0]
        type_num.append(poi_type_num)
    type_num_max = max(type_num)
    if type_num_max==0:type_num_max = 1
    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        parcel_area = round(
            gpd.GeoDataFrame(geometry=[parcel.geometry], crs='epsg:4326').to_crs('epsg:5234').area[0] / 1e6, 4)
        poi_intersection = gpd.overlay(poi, gpd.GeoDataFrame(geometry=[parcel.geometry], crs='epsg:4326'),
                                       how='intersection')
        poi_type_num = poi_intersection['type'].unique().shape[0]
        different_type_attraction = 0
        for j in range(poi_type_num):
            j_type_num = poi_intersection[poi_intersection['type']==poi_intersection['type'].unique()[j]].shape[0]
            weight = attraction_weight[poi_intersection['type'].unique()[j]]
            different_type_attraction += weight*j_type_num
        parcel_da = ((poi_type_num/(parcel_area*type_num_max))*different_type_attraction)/(predict_time-current_time)
        develop_attraction.append(parcel_da)


    land_file['DA'] = develop_attraction
    return land_file


def calculate_spatial_attraction_single_parcel(land_file,current_time,predict_time):
    '''
    空间吸引力
    :param land_file:
    :param current_time:
    :param predict_time:
    :return:
    '''
    spatial_attraction = []
    parcel_num = land_file.shape[0]
    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        # 计算地块中心点与曹庄地铁站距离
        parcel_centroid = parcel['geometry'].centroid
        parcel_distances = []
        for j in range(land_file.shape[0]):
            if j==i: continue
            other_parcel = land_file.iloc[j]
            other_parcel_centroid = other_parcel['geometry'].centroid
            data_p_o_centroid = gpd.GeoDataFrame({'geometry':[parcel_centroid,other_parcel_centroid]},crs='epsg:4326').to_crs('epsg:5234')
            distance = round(data_p_o_centroid.distance(data_p_o_centroid.shift())[1] / 1000, 4)
            parcel_distances.append(distance)
        parcel_distances = sum(parcel_distances)
        sa = ((2*((parcel_distances/(parcel_num-1))-1))/(parcel_num-2))/(predict_time-current_time)
        spatial_attraction.append(sa)
    land_file['SA'] = spatial_attraction
    return land_file

def calculate_develop_attraction_single_parcel(land_file,poi,current_time,predict_time):
    '''
    发展吸引力
    :param land_file:
    :param poi:
    :param current_time:
    :param predict_time:
    :return:
    '''
    attraction_weight = {'美食':0.1,'休闲娱乐':0.2,'酒店':0.1,'生活设施':0.2,'医疗':0.5,'公司':0.2,'体育文化':0.1,'教育':0.5}
    develop_attraction = []

    type_num = []
    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        poi_intersection = gpd.overlay(poi, gpd.GeoDataFrame(geometry=[parcel.geometry], crs='epsg:4326'),
                                       how='intersection')
        poi_type_num = poi_intersection['type'].unique().shape[0]
        type_num.append(poi_type_num)
    type_num_max = max(type_num)
    for i in tqdm(range(land_file.shape[0])):
        parcel = land_file.iloc[i]
        parcel_area = round(
            gpd.GeoDataFrame(geometry=[parcel.geometry], crs='epsg:4326').to_crs('epsg:5234').area[0] / 1e6, 4)
        poi_intersection = gpd.overlay(poi, gpd.GeoDataFrame(geometry=[parcel.geometry], crs='epsg:4326'),
                                       how='intersection')
        poi_type_num = poi_intersection['type'].unique().shape[0]
        different_type_attraction = 0
        for j in range(poi_type_num):
            j_type_num = poi_intersection[poi_intersection['type']==poi_intersection['type'].unique()[j]].shape[0]
            weight = attraction_weight[poi_intersection['type'].unique()[j]]
            different_type_attraction += weight*j_type_num
        parcel_da = ((poi_type_num/(parcel_area*type_num_max))*different_type_attraction)/(predict_time-current_time)
        develop_attraction.append(parcel_da)


    land_file['DA'] = develop_attraction
    return land_file
def random_to_5_or_0():
    a = random.randint(0,10)
    if a%2 == 0: return 5
    else:return 0

def find_nearest_edge(parcel_node, road_network):
    # 计算地块节点到所有道路节点的距离
    parcel_node = parcel_node.centroid
    distances = []
    for i in range(road_network.shape[0]):
        road_nodes = road_network.iloc[i]
        road_nodes = LineString(list(road_nodes.geometry.coords))
        distances.append(parcel_node.distance(road_nodes).values[0])
    # 找到最近节点的索引
    nearest_node_index = distances.index(min(distances))
    return nearest_node_index
def find_nearest_node(parcel_node, road_nodes):
    # 计算地块节点到所有道路节点的距离
    parcel_nodes = list(parcel_node.exterior.coords)
    road_nodes = LineString(list(road_nodes.geometry.coords))
    distances = []
    for node in parcel_nodes:
        node = Point(node)
        distances.append(node.distance(road_nodes))
    # 找到最近节点的索引
    nearest_node_index = distances.index(max(distances))

    return parcel_nodes[nearest_node_index]
def land_simulation_CA(land_file,current_time,predict_time,station_point,road_network_path,road_network_path_osm,poi):

    land_file_changing = land_file.copy()
    changing_threhold = 0.8
    road_network_changing = []
    road_network_gpd = gpd.read_file(road_network_path).to_crs(land_file_changing.crs)
    road_network_osm = ox.project_graph(ox.graph_from_xml(road_network_path_osm),to_crs=land_file_changing.crs)

    node_list = []
    node_from_s = []
    node_to_s = []

    for n in range(road_network_gpd.shape[0]):
        for nn in list(road_network_gpd.iloc[n]['geometry'].coords):
            node_list.append(nn)
    node_list = list(set(node_list))
    node_id = {elem: -1 - idx for idx, elem in enumerate(node_list)}
    for n in range(road_network_gpd.shape[0]):
        road_network_gpd_sub_coords = list(road_network_gpd.iloc[n]['geometry'].coords)
        node_from = road_network_gpd_sub_coords[0]
        node_from_id = node_id[node_from]
        node_to = road_network_gpd_sub_coords[1]
        node_to_id = node_id[node_to]
        node_from_s.append(node_from_id)
        node_to_s.append(node_to_id)
    road_network_gpd['from_'] = node_from_s
    road_network_gpd['to'] = node_to_s

    adding_node_from = 0
    adding_node_to = 10000
    for i in range(predict_time-current_time):
        # i时刻空间吸引力和发展吸引力
        # land_file_changing_sa = calculate_spatial_attraction(land_file_changing,i,i+1)
        # land_file_changing_da = calculate_develop_attraction(land_file_changing_sa,poi,i,i+1)
        # 上一时间路网演化
        if i == 0:
            pass
        else:
            for road_changing_item in road_network_changing:
                node1,node1y,node1x = road_changing_item[0][0],road_changing_item[0][1],road_changing_item[0][2]
                node2,node2y,node2x = road_changing_item[1][0],road_changing_item[1][1],road_changing_item[1][2]
                # road_network_osm.add_node(node1, y=node1x, x=node1y)
                # road_network_osm.add_node(node2, y=node2x, x=node2y)
                # road_network_osm.add_edge(node2,node1, osmid=-48, oneway=False)
                road_network_gpd_adding_edge = road_network_gpd.iloc[0]
                road_network_gpd_adding_edge_linestring = LineString([(node1x,node1y),(node2x,node2y)])
                from_ = node1
                to_ = node2
                road_network_gpd_adding_edge['geometry'] = road_network_gpd_adding_edge_linestring
                road_network_gpd_adding_edge['from_'] = from_
                road_network_gpd_adding_edge['to'] = to_
                road_network_gpd_adding_edge = {
                    'FID_2005_c':'add', 'DLMC':'add', 'newID':'add',
                'geometry':road_network_gpd_adding_edge_linestring,
                'from_':from_, 'to':to_
                }
                road_network_gpd = road_network_gpd.append(road_network_gpd_adding_edge,ignore_index=True)
            ox.save_graph_shapefile(road_network_osm, str(current_time+i)+'road')
            road_network_gpd.to_file(str(current_time+i)+'.shp')
            # road_network_gpd = gpd.read_file(str(current_time+i)+'road/edges.shp')
        # 当前时间路网生长
        road_network_changing = []
        for j in tqdm(range(land_file_changing.shape[0])):
            # 随机设置地块转换
            probability = random.random()
            if i == 0 and probability < 0.3:
                parcel = land_file_changing.iloc[j]
                # 地块状态，是否激活
                parcel_status = False
                # 三个指数转换潜力
                parcel_potential = parcel['POTENTIA_1']
                # 地块类型
                parcel_type = parcel['newID']
                # 待转地块类型
                parcel_changing = [0,1,2,3,4,5,6]
                parcel_changing.remove(parcel_type)
                parcel_changing_posibility = {}
                for parcel_changing_item in parcel_changing:
                    parcel_changing_posibility[parcel_changing_item] = 0.1
                # 领域效应
                parcel_touches = land_file_changing[land_file_changing.touches(parcel.geometry)]
                parcel_touches_road_intersection = gpd.sjoin(parcel_touches,road_network_gpd)
                parcel_potential += parcel_touches_road_intersection.shape[0]/road_network_gpd.shape[0]
                parcel_touches_type = parcel_touches['newID'].unique()
                for parcel_changing_item in parcel_changing:
                    if parcel_changing_item == 3:
                        parcel_changing_posibility[parcel_changing_item] = 0.000001
                        continue
                    # 基础转换概率
                    basic_posibility = parcel_potential
                    # 当前类型为商业用地，
                    if parcel_type == 0:
                        changing_posibility = basic_posibility * 0.5
                        # 转换对象为商业用地或者居住用地
                        # if parcel_changing_item in [5]:
                        #     if 5 in parcel_touches_type or 0 in parcel_touches_type:
                        #         changing_posibility = basic_posibility*1.8
                        #     else:
                        #         changing_posibility = basic_posibility*1.2
                        # elif parcel_changing_item in [0,2]:
                        #     changing_posibility = basic_posibility
                        # # 若转换对象为绿地或者公共服务设施
                        # # if parcel_changing_item in [2,1]:
                        # else:
                        #     changing_posibility = basic_posibility*0.6
                    # 当前类型为居住用地
                    elif parcel_type == 5:
                        if parcel_changing_item in [5]:
                            if 5 in parcel_touches_type or 0 in parcel_touches_type:
                                changing_posibility = basic_posibility * 1.8
                            else:
                                changing_posibility = basic_posibility*1.2
                        elif parcel_changing_item in [2]:
                            changing_posibility = basic_posibility
                        # 若转换对象为绿地或者工业用地
                        # if parcel_changing_item in [2, 1]:
                        else:
                            changing_posibility = basic_posibility * 0.5
                    # 当前类型为待开发用地
                    elif parcel_type == 3:
                        if parcel_changing_item in [5]:
                            if 5 in parcel_touches_type or 0 in parcel_touches_type or 4 in parcel_touches_type:
                                changing_posibility = basic_posibility * 3
                            elif 2 in parcel_touches_type:
                                changing_posibility = basic_posibility * 1.2
                            else:changing_posibility = basic_posibility
                        elif parcel_changing_item in [6]:
                            if 5 in parcel_touches_type or 0 in parcel_touches_type or 4 in parcel_touches_type:
                                changing_posibility = basic_posibility * 2
                            elif 2 in parcel_touches_type:
                                changing_posibility = basic_posibility * 1.2
                            else:changing_posibility = basic_posibility
                        elif parcel_changing_item in [2]:
                            if 5 in parcel_touches_type:
                                changing_posibility = basic_posibility * 0.8
                            else:
                                changing_posibility = basic_posibility
                        else:
                            changing_posibility = basic_posibility * 0.6
                    # 当前类型为工业用地
                    elif parcel_type == 2:
                        changing_posibility = basic_posibility * 0.5
                        # if parcel_changing_item in [0, 5, 4]:
                        #     # 若领域为居住商业公共服务
                        #     if 5 in parcel_touches_type or 0 in parcel_touches_type or 4 in parcel_touches_type:
                        #         changing_posibility = basic_posibility*0.6
                        #     else:
                        #         changing_posibility = basic_posibility
                        # elif parcel_changing_item ==1:
                        #     changing_posibility = basic_posibility
                        # else:
                        #     changing_posibility = basic_posibility * 1.2
                    # 当前类型为绿色空间
                    elif parcel_type == 1:
                        changing_posibility = basic_posibility * 0.001
                    # 当前类型为公共服务用地
                    else:
                        changing_posibility = basic_posibility * 0.5
                    # elif parcel_type == 4:
                    #     if parcel_changing_item in [0,5,4]:
                    #         if 5 in parcel_touches_type or 0 in parcel_touches_type:
                    #             changing_posibility = basic_posibility * 1.8
                    #         else:
                    #             changing_posibility = basic_posibility*1.2
                    #     # 若转换对象为绿地或者工业用地
                    #     # if parcel_changing_item in [2, 1]:
                    #     else:
                    #         changing_posibility = basic_posibility * 0.6
                    parcel_changing_posibility[parcel_changing_item] = changing_posibility

                # 圈层效应
                parcel_centroid = parcel['geometry'].centroid
                data_cen_caozhuang = gpd.GeoDataFrame({'geometry': [parcel_centroid, station_point]},
                                                      crs='epsg:4326').to_crs('epsg:5234')
                distance = round(data_cen_caozhuang.distance(data_cen_caozhuang.shift())[1] / 1000, 3)
                for parcel_changing_item_2 in parcel_changing:
                    changing_posibility2 = parcel_changing_posibility[parcel_changing_item_2]
                    if distance < 0.5:
                        if parcel_changing_item_2 == 0:
                            changing_posibility2*=1.5
                        elif parcel_changing_item_2 == 5:
                            changing_posibility2*=1.2
                        else:changing_posibility2 = changing_posibility2
                    elif (distance > 0.5 and distance < 1.0):
                        if parcel_changing_item_2 == 5:
                            changing_posibility2*=1.3
                        elif parcel_changing_item_2 == 0:
                            changing_posibility2*=1.1
                        else:changing_posibility2 = changing_posibility2
                    else:
                        if parcel_changing_item_2 == 2:
                            changing_posibility2*=1.2
                        else:
                            changing_posibility2 = changing_posibility2
                    parcel_changing_posibility[parcel_changing_item_2] = changing_posibility2
                # 转换概率最高的目标类型
                max_changing_posibility_type = max(parcel_changing_posibility, key=lambda x: parcel_changing_posibility[x])
                max_changing_posibility_name = newID_DLMC[max_changing_posibility_type][0]
                max_changing_posibility = parcel_changing_posibility[max_changing_posibility_type]
                if max_changing_posibility > changing_threhold:
                    land_file_changing.iloc[j,0] = max_changing_posibility_name
                    land_file_changing.iloc[j,1] = max_changing_posibility_type
                    parcel_status = True
                    parcel_touches_sa = calculate_spatial_attraction(parcel_touches,i,i+1)
                    parcel_touches_da = calculate_develop_attraction(parcel_touches_sa,poi,i,i+1)
                    parcel_touches_sd = parcel_touches_da
                    parcel_touches_sd['SD'] = parcel_touches_sd['SA'] + parcel_touches_sd['DA']
                    # 综合吸引力最大的地块
                    parcel_touches_sd_max = parcel_touches_sd[parcel_touches_sd['SD']==parcel_touches_sd['SD'].max()]
                    # 路网相交部分
                    road_parcel_max_intersection = gpd.sjoin(parcel_touches_sd_max,road_network_gpd,how='inner',op='intersects')
                    # 公共边
                    try:
                        # # 距离最近的路网边
                        # nearest_edge = find_nearest_edge(parcel_touches_sd_max,road_network_gpd)
                        # parcel_road_nearse_point = nearest_points(parcel_touches_sd_max.geometry.unary_union,
                        #                                           road_network_gpd.unary_union)
                        # # 包含路网点的边
                        # contain_mask = road_network_gpd.geometry.intersects(parcel_road_nearse_point[0])
                        # road_network_contains = road_network_gpd[contain_mask]
                        # for road_network_contain_index in range(road_network_contains.shape[0]):
                        #     road_network_contain = road_network_contains.iloc[road_network_contain_index]
                        #     # 距离这个边最近的地块点
                        #     road_network_contain_nearst_point = parcel_touches_sd_max['geometry'].apply(
                        #         lambda x: find_nearest_node(x, road_network_contain)).iloc[0]
                        #     road_network_contain_nearst_point = Point(road_network_contain_nearst_point)
                        #     projection = road_network_contain.geometry.interpolate(
                        #         road_network_contain.geometry.project(road_network_contain_nearst_point))
                        #     intersection = road_network_contain.geometry.intersection(projection)
                        #     road_add_from_y = road_network_contain_nearst_point.y
                        #     road_add_from_x = road_network_contain_nearst_point.x
                        #     road_add_from_id = adding_node_from + 1
                        #     try:
                        #         road_add_to_y = intersection.y
                        #         road_add_to_x = intersection.x
                        #         road_add_to_id = adding_node_to + 1
                        #     except:
                        #         road_add_to_y = projection.y
                        #         road_add_to_x = projection.x
                        #         road_add_to_id = adding_node_to + 1
                        #     road_network_changing.append([[road_add_from_id, road_add_from_y, road_add_from_x],[road_add_to_id, road_add_to_y, road_add_to_x]])
                        #     adding_node_from +=1
                        #     adding_node_to +=1
                        #     print('road adding')
                        # 距离最近的路网边
                        nearest_edge_index = find_nearest_edge(parcel_touches_sd_max, road_network_gpd)
                        nearest_edge = road_network_gpd.iloc[nearest_edge_index]
                        parcel_road_nearse_point = nearest_points(parcel_touches_sd_max.geometry.unary_union,
                                                                  road_network_gpd.unary_union)
                        # 距离这个边最近的地块点
                        road_network_contain_nearst_point = parcel_touches_sd_max['geometry'].apply(
                            lambda x: find_nearest_node(x, nearest_edge)).iloc[0]
                        road_network_contain_nearst_point = Point(road_network_contain_nearst_point)
                        # 整个地块中距离这个边最近的点
                        # whole_parcel_nearst_point = land_file_changing['geometry'].apply(
                        #     lambda x: find_nearest_node(x, nearest_edge)).iloc[0]
                        # whole_parcel_nearst_point = Point(whole_parcel_nearst_point)
                        # G = nx.Graph()
                        # node_index = 0
                        # for index_a, row in land_file_changing.iterrows():
                        #     for index_b,boundary_item in enumerate(list(row['geometry'].boundary.coords)):
                        #         if index_b == len(list(row['geometry'].boundary.coords))-1:
                        #             line_geometry = LineString((list(row['geometry'].boundary.coords)[index_b],
                        #                                         list(row['geometry'].boundary.coords)[0]))
                        #             G.add_node(node_index,pos=row['geometry'].boundary.coords[index_b])
                        #             G.add_node(node_index+1,
                        #                        pos=row['geometry'].boundary.coords[0])
                        #
                        #             G.add_edge(node_index, node_index+1,
                        #                        geometry=line_geometry)
                        #             if road_network_contain_nearst_point.coords[0] == row['geometry'].boundary.coords[index_b]:
                        #                 start_node = node_index
                        #             if whole_parcel_nearst_point.coords[0] == row['geometry'].boundary.coords[0]:
                        #                 end_node = node_index+1
                        #             node_index += 1
                        #             break
                        #         line_geometry = LineString((list(row['geometry'].boundary.coords)[index_b],list(row['geometry'].boundary.coords)[index_b+1]))
                        #         G.add_node(node_index, pos=list(row['geometry'].boundary.coords)[index_b])
                        #         G.add_node(node_index + 1,
                        #                    pos=list(row['geometry'].boundary.coords)[index_b+1])
                        #         if road_network_contain_nearst_point.coords[0] == list(row['geometry'].boundary.coords)[index_b]:
                        #             start_node = node_index
                        #         if whole_parcel_nearst_point.coords[0] == list(row['geometry'].boundary.coords)[index_b+1]:
                        #             end_node = node_index + 1
                        #         node_index += 1
                        #         G.add_edge(node_index,node_index + 1,geometry=line_geometry)
                        # shortest_path_nodes = nx.shortest_path(G, source=start_node, target=end_node)

                        # projection = nearest_edge.geometry.interpolate(
                        #     nearest_edge.geometry.project(road_network_contain_nearst_point))
                        # intersection = nearest_edge.geometry.intersection(projection)
                        for point_index,point_parcel_sdmax in enumerate(list(parcel_touches_sd_max['geometry'].boundary.iloc[0].coords)):
                            if point_index == len(list(parcel_touches_sd_max['geometry'].boundary.iloc[0].coords))-1:
                                road_add_from_y = point_parcel_sdmax[1]
                                road_add_from_x = point_parcel_sdmax[0]
                                road_add_from_id = adding_node_from + 1

                                road_add_to_y = \
                                list(parcel_touches_sd_max['geometry'].iloc[0].boundary.coords[0])[1]
                                road_add_to_x = \
                                list(parcel_touches_sd_max['geometry'].iloc[0].boundary.coords[0])[0]
                                road_add_to_id = adding_node_to + 1
                                road_network_changing.append([[road_add_from_id, road_add_from_y, road_add_from_x],
                                                              [road_add_to_id, road_add_to_y, road_add_to_x]])
                                adding_node_from += 1
                                adding_node_to += 1
                                print('road adding')
                                break
                            road_add_from_y = point_parcel_sdmax[1]
                            road_add_from_x = point_parcel_sdmax[0]
                            road_add_from_id = adding_node_from + 1

                            road_add_to_y = list(parcel_touches_sd_max['geometry'].iloc[0].boundary.coords[point_index+1])[1]
                            road_add_to_x = list(parcel_touches_sd_max['geometry'].iloc[0].boundary.coords[point_index+1])[0]
                            road_add_to_id = adding_node_to + 1
                            road_network_changing.append([[road_add_from_id, road_add_from_y, road_add_from_x],
                                                          [road_add_to_id, road_add_to_y, road_add_to_x]])
                            adding_node_from += 1
                            adding_node_to += 1
                            print('road adding')
                        # for point_edge in list(nearest_edge.geometry.coords):
                        #     road_add_from_y = road_network_contain_nearst_point.y
                        #     road_add_from_x = road_network_contain_nearst_point.x
                        #     road_add_from_id = adding_node_from + 1
                        #
                        #     road_add_to_y = point_edge[1]
                        #     road_add_to_x = point_edge[0]
                        #     road_add_to_id = adding_node_to + 1
                        #
                        #     # 包含路网点的边
                        #     # contain_mask = road_network_gpd.geometry.intersects(parcel_road_nearse_point[0])
                        #     # road_network_contains = road_network_gpd[contain_mask]
                        #     # for road_network_contain_index in range(road_network_contains.shape[0]):
                        #     #     road_network_contain = road_network_contains.iloc[road_network_contain_index]
                        #     #
                        #     #     # 距离这个边最近的地块点
                        #     #     road_network_contain_nearst_point = parcel_touches_sd_max['geometry'].apply(
                        #     #         lambda x: find_nearest_node(x, road_network_contain)).iloc[0]
                        #     #     road_network_contain_nearst_point = Point(road_network_contain_nearst_point)
                        #     #     projection = road_network_contain.geometry.interpolate(
                        #     #         road_network_contain.geometry.project(road_network_contain_nearst_point))
                        #     #     intersection = road_network_contain.geometry.intersection(projection)
                        #     #     road_add_from_y = road_network_contain_nearst_point.y
                        #     #     road_add_from_x = road_network_contain_nearst_point.x
                        #     #     road_add_from_id = adding_node_from + 1
                        #     #     try:
                        #     #         road_add_to_y = intersection.y
                        #     #         road_add_to_x = intersection.x
                        #     #         road_add_to_id = adding_node_to + 1
                        #     #     except:
                        #     #         road_add_to_y = projection.y
                        #     #         road_add_to_x = projection.x
                        #     #         road_add_to_id = adding_node_to + 1
                        #     road_network_changing.append([[road_add_from_id, road_add_from_y, road_add_from_x],
                        #                                   [road_add_to_id, road_add_to_y, road_add_to_x]])
                        #     adding_node_from += 1
                        #     adding_node_to += 1
                        #     print('road adding')
                    except:
                        print('wrong one')

            else:
                parcel = land_file_changing.iloc[j]
                # 地块状态，是否激活
                parcel_status = False
                # 三个指数转换潜力
                parcel_potential = parcel['POTENTIA_1']
                # 地块类型
                parcel_type = parcel['newID']
                # 待转地块类型
                parcel_changing = [0, 1, 2, 3, 4, 5,6]
                parcel_changing.remove(parcel_type)
                parcel_changing_posibility = {}
                for parcel_changing_item in parcel_changing:
                    parcel_changing_posibility[parcel_changing_item] = 0.1
                # 领域效应
                parcel_touches = land_file_changing[land_file_changing.touches(parcel.geometry)]
                parcel_touches_type = parcel_touches['newID'].unique()
                for parcel_changing_item in parcel_changing:
                    if parcel_changing_item == 3:
                        parcel_changing_posibility[parcel_changing_item] = 0.000001
                        continue
                    # 基础转换概率
                    basic_posibility = parcel_potential
                    # 当前类型为商业用地，
                    if parcel_type == 0:
                        changing_posibility = basic_posibility * 0.5
                        # 转换对象为商业用地或者居住用地
                        # if parcel_changing_item in [5]:
                        #     if 5 in parcel_touches_type or 0 in parcel_touches_type:
                        #         changing_posibility = basic_posibility*1.8
                        #     else:
                        #         changing_posibility = basic_posibility*1.2
                        # elif parcel_changing_item in [0,2]:
                        #     changing_posibility = basic_posibility
                        # # 若转换对象为绿地或者公共服务设施
                        # # if parcel_changing_item in [2,1]:
                        # else:
                        #     changing_posibility = basic_posibility*0.6
                    # 当前类型为居住用地
                    elif parcel_type == 5:
                        if parcel_changing_item in [5]:
                            if 5 in parcel_touches_type or 0 in parcel_touches_type:
                                changing_posibility = basic_posibility * 1.8
                            else:
                                changing_posibility = basic_posibility * 1.2
                        elif parcel_changing_item in [2]:
                            changing_posibility = basic_posibility
                        # 若转换对象为绿地或者工业用地
                        # if parcel_changing_item in [2, 1]:
                        else:
                            changing_posibility = basic_posibility * 0.5
                    # 当前类型为待开发用地
                    elif parcel_type == 3:
                        if parcel_changing_item in [5]:
                            if 5 in parcel_touches_type or 0 in parcel_touches_type or 4 in parcel_touches_type:
                                changing_posibility = basic_posibility * 3
                            elif 2 in parcel_touches_type:
                                changing_posibility = basic_posibility * 0.8
                            else:
                                changing_posibility = basic_posibility
                        if parcel_changing_item in [6]:
                            if 5 in parcel_touches_type or 0 in parcel_touches_type or 4 in parcel_touches_type:
                                changing_posibility = basic_posibility * 2.5
                            elif 2 in parcel_touches_type:
                                changing_posibility = basic_posibility * 0.8
                            else:
                                changing_posibility = basic_posibility
                        elif parcel_changing_item in [2]:
                            if 5 in parcel_touches_type:
                                changing_posibility = basic_posibility * 0.8
                            else:
                                changing_posibility = basic_posibility * 1.2
                        else:
                            changing_posibility = basic_posibility * 0.6
                    # 当前类型为工业用地
                    elif parcel_type == 2:
                        changing_posibility = basic_posibility * 0.5
                        # if parcel_changing_item in [0, 5, 4]:
                        #     # 若领域为居住商业公共服务
                        #     if 5 in parcel_touches_type or 0 in parcel_touches_type or 4 in parcel_touches_type:
                        #         changing_posibility = basic_posibility*0.6
                        #     else:
                        #         changing_posibility = basic_posibility
                        # elif parcel_changing_item ==1:
                        #     changing_posibility = basic_posibility
                        # else:
                        #     changing_posibility = basic_posibility * 1.2
                    # 当前类型为绿色空间
                    elif parcel_type == 1:
                        changing_posibility = basic_posibility * 0.001
                    # 当前类型为公共服务用地
                    else:
                        changing_posibility = basic_posibility * 0.5
                    # elif parcel_type == 4:
                    #     if parcel_changing_item in [0,5,4]:
                    #         if 5 in parcel_touches_type or 0 in parcel_touches_type:
                    #             changing_posibility = basic_posibility * 1.8
                    #         else:
                    #             changing_posibility = basic_posibility*1.2
                    #     # 若转换对象为绿地或者工业用地
                    #     # if parcel_changing_item in [2, 1]:
                    #     else:
                    #         changing_posibility = basic_posibility * 0.6
                    parcel_changing_posibility[parcel_changing_item] = changing_posibility

                # 圈层效应
                parcel_centroid = parcel['geometry'].centroid
                data_cen_caozhuang = gpd.GeoDataFrame({'geometry': [parcel_centroid, station_point]},
                                                      crs='epsg:4326').to_crs('epsg:5234')
                distance = round(data_cen_caozhuang.distance(data_cen_caozhuang.shift())[1] / 1000, 3)
                for parcel_changing_item_2 in parcel_changing:
                    changing_posibility2 = parcel_changing_posibility[parcel_changing_item_2]
                    if distance < 0.5:
                        if parcel_changing_item_2 == 0:
                            changing_posibility2 *= 1.5
                        elif parcel_changing_item_2 == 5:
                            changing_posibility2 *= 1.2
                        else:
                            changing_posibility2 = changing_posibility2
                    elif (distance > 0.5 and distance < 1.0):
                        if parcel_changing_item_2 == 5:
                            changing_posibility2 *= 1.3
                        elif parcel_changing_item_2 == 0:
                            changing_posibility2 *= 1.1
                        else:
                            changing_posibility2 = changing_posibility2
                    else:
                        if parcel_changing_item_2 == 2:
                            changing_posibility2 *= 1.2
                        else:
                            changing_posibility2 = changing_posibility2
                    parcel_changing_posibility[parcel_changing_item_2] = changing_posibility2
                # 转换概率最高的目标类型
                max_changing_posibility_type = max(parcel_changing_posibility,
                                                   key=lambda x: parcel_changing_posibility[x])
                max_changing_posibility_name = newID_DLMC[max_changing_posibility_type][0]
                max_changing_posibility = parcel_changing_posibility[max_changing_posibility_type]
                if max_changing_posibility > changing_threhold:
                    land_file_changing.iloc[j, 0] = max_changing_posibility_name
                    land_file_changing.iloc[j, 1] = max_changing_posibility_type
                    parcel_status = True
                    parcel_touches_sa = calculate_spatial_attraction(parcel_touches,i,i+1)
                    parcel_touches_da = calculate_develop_attraction(parcel_touches_sa,poi,i,i+1)
                    parcel_touches_sd = parcel_touches_da
                    parcel_touches_sd['SD'] = parcel_touches_sd['SA'] + parcel_touches_sd['DA']
                    # 综合吸引力最大的地块
                    parcel_touches_sd_max = parcel_touches_sd[parcel_touches_sd['SD']==parcel_touches_sd['SD'].max()]
                    # 路网相交部分
                    # road_parcel_max_intersection = gpd.sjoin(parcel_touches_sd_max,road_network_gpd,how='inner',op='intersects')
                    # 公共边
                    try:
                        nearest_edge_index = find_nearest_edge(parcel_touches_sd_max, road_network_gpd)
                        nearest_edge = road_network_gpd.iloc[nearest_edge_index]
                        parcel_road_nearse_point = nearest_points(parcel_touches_sd_max.geometry.unary_union,
                                                                  road_network_gpd.unary_union)
                        # 距离这个边最近的地块点
                        road_network_contain_nearst_point = parcel_touches_sd_max['geometry'].apply(
                            lambda x: find_nearest_node(x, nearest_edge)).iloc[0]
                        road_network_contain_nearst_point = Point(road_network_contain_nearst_point)
                        # projection = nearest_edge.geometry.interpolate(
                        #     nearest_edge.geometry.project(road_network_contain_nearst_point))
                        # intersection = nearest_edge.geometry.intersection(projection)
                        for point_index,point_parcel_sdmax in enumerate(list(parcel_touches_sd_max['geometry'].boundary.iloc[0].coords)):
                            if point_index == len(list(parcel_touches_sd_max['geometry'].boundary.iloc[0].coords))-1:
                                road_add_from_y = point_parcel_sdmax[1]
                                road_add_from_x = point_parcel_sdmax[0]
                                road_add_from_id = adding_node_from + 1

                                road_add_to_y = \
                                list(parcel_touches_sd_max['geometry'].iloc[0].boundary.coords[0])[1]
                                road_add_to_x = \
                                list(parcel_touches_sd_max['geometry'].iloc[0].boundary.coords[0])[0]
                                road_add_to_id = adding_node_to + 1
                                road_network_changing.append([[road_add_from_id, road_add_from_y, road_add_from_x],
                                                              [road_add_to_id, road_add_to_y, road_add_to_x]])
                                adding_node_from += 1
                                adding_node_to += 1
                                print('road adding')
                                break
                            road_add_from_y = point_parcel_sdmax[1]
                            road_add_from_x = point_parcel_sdmax[0]
                            road_add_from_id = adding_node_from + 1

                            road_add_to_y = list(parcel_touches_sd_max['geometry'].iloc[0].boundary.coords[point_index+1])[1]
                            road_add_to_x = list(parcel_touches_sd_max['geometry'].iloc[0].boundary.coords[point_index+1])[0]
                            road_add_to_id = adding_node_to + 1
                            road_network_changing.append([[road_add_from_id, road_add_from_y, road_add_from_x],
                                                          [road_add_to_id, road_add_to_y, road_add_to_x]])
                            adding_node_from += 1
                            adding_node_to += 1
                            print('road adding')
                        # for point_edge in list(nearest_edge.geometry.coords):
                        #     road_add_from_y = road_network_contain_nearst_point.y
                        #     road_add_from_x = road_network_contain_nearst_point.x
                        #     road_add_from_id = adding_node_from + 1
                        #
                        #     road_add_to_y = point_edge[1]
                        #     road_add_to_x = point_edge[0]
                        #     road_add_to_id = adding_node_to + 1
                        #
                        #     # 包含路网点的边
                        #     # contain_mask = road_network_gpd.geometry.intersects(parcel_road_nearse_point[0])
                        #     # road_network_contains = road_network_gpd[contain_mask]
                        #     # for road_network_contain_index in range(road_network_contains.shape[0]):
                        #     #     road_network_contain = road_network_contains.iloc[road_network_contain_index]
                        #     #
                        #     #     # 距离这个边最近的地块点
                        #     #     road_network_contain_nearst_point = parcel_touches_sd_max['geometry'].apply(
                        #     #         lambda x: find_nearest_node(x, road_network_contain)).iloc[0]
                        #     #     road_network_contain_nearst_point = Point(road_network_contain_nearst_point)
                        #     #     projection = road_network_contain.geometry.interpolate(
                        #     #         road_network_contain.geometry.project(road_network_contain_nearst_point))
                        #     #     intersection = road_network_contain.geometry.intersection(projection)
                        #     #     road_add_from_y = road_network_contain_nearst_point.y
                        #     #     road_add_from_x = road_network_contain_nearst_point.x
                        #     #     road_add_from_id = adding_node_from + 1
                        #     #     try:
                        #     #         road_add_to_y = intersection.y
                        #     #         road_add_to_x = intersection.x
                        #     #         road_add_to_id = adding_node_to + 1
                        #     #     except:
                        #     #         road_add_to_y = projection.y
                        #     #         road_add_to_x = projection.x
                        #     #         road_add_to_id = adding_node_to + 1
                        #     road_network_changing.append([[road_add_from_id, road_add_from_y, road_add_from_x],
                        #                                   [road_add_to_id, road_add_to_y, road_add_to_x]])
                        #     adding_node_from += 1
                        #     adding_node_to += 1
                        #     print('road adding')
                    except:
                        print('wrong one')

    return land_file_changing


if __name__ == '__main__':
    path_2005 = r'vector_new/2005_split.shp'
    path_2008 = r'vector_new/2008_split.shp'
    path_2011 = r'vector_new/2011_split.shp'
    path_2014 = r'vector_new/2014_split.shp'
    path_2017 = r'vector_new/2017_split.shp'
    path_2020 = r'vector_new/2020_split.shp'
    path_2023 = r'vector_new/2023_split.shp'




    # 计算土地利用扩散衰减指数LDAI
    # shp_2005,shp_2008,shp_2011,shp_2014,shp_2017,shp_2020,shp_2023 = gpd.read_file(path_2005),gpd.read_file(path_2008),gpd.read_file(path_2011),\
    #                                                                   gpd.read_file(path_2014),gpd.read_file(path_2017),gpd.read_file(path_2020),\
    #                                                                   gpd.read_file(path_2023)
    #
    # shp_2005,shp_2008,shp_2011,shp_2014,shp_2017,shp_2020,shp_2023 = shp_2005.to_crs({'init': 'epsg:4326'}),shp_2008.to_crs({'init': 'epsg:4326'}),\
    #                                                                  shp_2011.to_crs({'init': 'epsg:4326'}),shp_2014.to_crs({'init': 'epsg:4326'}),\
    #                                                                  shp_2017.to_crs({'init': 'epsg:4326'}),shp_2020.to_crs({'init': 'epsg:4326'}),\
    #                                                                  shp_2023.to_crs({'init': 'epsg:4326'})
    #
    # caozhuang_station = Point(117.071487,39.151539)
    # shp_2005_LDAI = calculate_land_diffuse_attenuation_index(shp_2005,caozhuang_station).to_file(r'LDAI/2005_split_ldai.shp')
    # shp_2008_LDAI = calculate_land_diffuse_attenuation_index(shp_2008, caozhuang_station).to_file(r'LDAI/2008_split_ldai.shp')
    # shp_2011_LDAI = calculate_land_diffuse_attenuation_index(shp_2011, caozhuang_station).to_file(r'LDAI/2011_split_ldai.shp')
    # shp_2014_LDAI = calculate_land_diffuse_attenuation_index(shp_2014, caozhuang_station).to_file(r'LDAI/2014_split_ldai.shp')
    # shp_2017_LDAI = calculate_land_diffuse_attenuation_index(shp_2017, caozhuang_station).to_file(r'LDAI/2017_split_ldai.shp')
    # shp_2020_LDAI = calculate_land_diffuse_attenuation_index(shp_2020, caozhuang_station).to_file(r'LDAI/2020_split_ldai.shp')
    # shp_2023_LDAI = calculate_land_diffuse_attenuation_index(shp_2023, caozhuang_station).to_file(r'LDAI/2023_split_ldai.shp')



    # 计算土地资源价值指数LRVI
    # path_2005_ldai,path_2008_ldai,path_2011_ldai,\
    # path_2014_ldai,path_2017_ldai,path_2020_ldai,path_2023_ldai, = r'LDAI/2005_split_ldai.shp',r'LDAI/2008_split_ldai.shp',\
    #                                                                 r'LDAI/2011_split_ldai.shp',r'LDAI/2014_split_ldai.shp',\
    #                                                                r'LDAI/2017_split_ldai.shp',r'LDAI/2020_split_ldai.shp',\
    #                                                                r'LDAI/2023_split_ldai.shp'
    # shp_2005,shp_2008,shp_2011,shp_2014,shp_2017,shp_2020,shp_2023 = gpd.read_file(path_2005_ldai),gpd.read_file(path_2008_ldai),\
    #                                                                  gpd.read_file(path_2011_ldai),\
    #                                                                   gpd.read_file(path_2014_ldai),gpd.read_file(path_2017_ldai),\
    #                                                                  gpd.read_file(path_2020_ldai),\
    #                                                                   gpd.read_file(path_2023_ldai)
    # path_road = r'road_network/road_network.shp\edges.shp'
    # path_poi = r'poi_clip.shp'
    # road_network = gpd.read_file(path_road)
    # poi = gpd.read_file(path_poi)
    # shp_2005_LRVI = calculate_land_resource_value_index(shp_2005,road_network,poi).to_file(r'LRVI/2005_split.shp')
    # shp_2008_LRVI = calculate_land_resource_value_index(shp_2008,road_network,poi).to_file(r'LRVI/2008_split.shp')
    # shp_2011_LRVI = calculate_land_resource_value_index(shp_2011,road_network,poi).to_file(r'LRVI/2011_split.shp')
    # shp_2014_LRVI = calculate_land_resource_value_index(shp_2014,road_network,poi).to_file(r'LRVI/2014_split.shp')
    # shp_2017_LRVI = calculate_land_resource_value_index(shp_2017,road_network,poi).to_file(r'LRVI/2017_split.shp')
    # shp_2020_LRVI = calculate_land_resource_value_index(shp_2020,road_network,poi).to_file(r'LRVI/2020_split.shp')



    # 计算土地利用转化难度指数
    # path_2005_LRVI = r'LRVI/2005_split.shp'
    # shp_2005 = gpd.read_file(path_2005_LRVI)
    # shp_2005_LUCI = calculate_land_use_change_index(shp_2005).to_file(r'LUCI/2005_split.shp')
    # path_2008_LRVI = r'LRVI/2008_split.shp'
    # shp_2008 = gpd.read_file(path_2008_LRVI)
    # shp_2008_LUCI = calculate_land_use_change_index(shp_2008).to_file(r'LUCI/2008_split.shp')
    # path_2011_LRVI = r'LRVI/2011_split.shp'
    # shp_2011 = gpd.read_file(path_2011_LRVI)
    # shp_2011_LUCI = calculate_land_use_change_index(shp_2011).to_file(r'LUCI/2011_split.shp')
    # path_2014_LRVI = r'LRVI/2014_split.shp'
    # shp_2014 = gpd.read_file(path_2014_LRVI)
    # shp_2014_LUCI = calculate_land_use_change_index(shp_2014).to_file(r'LUCI/2014_split.shp')
    # path_2017_LRVI = r'LRVI/2017_split.shp'
    # shp_2017 = gpd.read_file(path_2017_LRVI)
    # shp_2017_LUCI = calculate_land_use_change_index(shp_2017).to_file(r'LUCI/2017_split.shp')
    # path_2020_LRVI = r'LRVI/2020_split.shp'
    # shp_2020 = gpd.read_file(path_2020_LRVI)
    # shp_2020_LUCI = calculate_land_use_change_index(shp_2020).to_file(r'LUCI/2020_split.shp')

    



    # 归一化融合
    # path_2005_LUCI = r'LUCI/2005_split.shp'
    # shp_2005 = gpd.read_file(path_2005_LUCI)
    # shp_2005 = nomalize_indexs(shp_2005).to_file(r'POTENTIAL_FINAL/2005_split.shp')
    # path_2008_LUCI = r'LUCI/2008_split.shp'
    # shp_2008 = gpd.read_file(path_2008_LUCI)
    # shp_2008 = nomalize_indexs(shp_2008).to_file(r'POTENTIAL_FINAL/2008_split.shp')
    # path_2011_LUCI = r'LUCI/2011_split.shp'
    # shp_2011 = gpd.read_file(path_2011_LUCI)
    # shp_2011 = nomalize_indexs(shp_2011).to_file(r'POTENTIAL_FINAL/2011_split.shp')
    # path_2014_LUCI = r'LUCI/2014_split.shp'
    # shp_2014 = gpd.read_file(path_2014_LUCI)
    # shp_2014 = nomalize_indexs(shp_2014).to_file(r'POTENTIAL_FINAL/2014_split.shp')
    # path_2017_LUCI = r'LUCI/2017_split.shp'
    # shp_2017 = gpd.read_file(path_2017_LUCI)
    # shp_2017 = nomalize_indexs(shp_2017).to_file(r'POTENTIAL_FINAL/2017_split.shp')
    # path_2020_LUCI = r'LUCI/2020_split.shp'
    # shp_2020 = gpd.read_file(path_2020_LUCI)
    # shp_2020 = nomalize_indexs(shp_2020).to_file(r'POTENTIAL_FINAL/2020_split.shp')



    # 计算空间吸引力和发展吸引力
    # path_2005_potential = r'POTENTIAL_FINAL/2005_split.shp'
    # shp_2005 = gpd.read_file(path_2005_potential)
    # path_road = r'road_network/road_network.shp\edges.shp'
    # path_poi = r'poi_clip.shp'
    # road_network = gpd.read_file(path_road)
    # poi = gpd.read_file(path_poi)
    # # shp_2005 = calculate_spatial_attraction(shp_2005,2005,2008)
    # shp_2005 = calculate_develop_attraction(shp_2005,poi,2005,2008)
    # path_2008_potential = r'POTENTIAL_FINAL/2008_split.shp'
    # shp_2008 = gpd.read_file(path_2008_potential)
    # path_road = r'road_network/road_network.shp\edges.shp'
    # path_poi = r'poi_clip.shp'
    # road_network = gpd.read_file(path_road)
    # poi = gpd.read_file(path_poi)
    # # shp_2005 = calculate_spatial_attraction(shp_2005,2005,2008)
    # shp_2008 = calculate_develop_attraction(shp_2008,poi,2008,2011)


    # path_2005_potential = r'POTENTIAL_FINAL/2005_split.shp'
    # path_road = r'2005.shp'
    # path_road_osm = r'2005.osm'
    # path_poi = r'/poi_clip.shp'
    # shp_2005 = gpd.read_file(path_2005_potential)
    # caozhuang_station = Point(117.071487, 39.151539)
    # poi = gpd.read_file(path_poi)
    # shp_2008_simulation = land_simulation_CA(shp_2005,2005,2012,caozhuang_station,path_road,path_road_osm,poi)
    # shp_2008_simulation.to_file(r'test/test.shp')
    path_2017_potential = r'POTENTIAL_FINAL/2017_split.shp'
    path_road = r'2017.shp'
    path_road_osm = r'2005.osm'
    path_poi = r'poi_clip.shp'
    shp_2017 = gpd.read_file(path_2017_potential)
    caozhuang_station = Point(117.071487, 39.151539)
    poi = gpd.read_file(path_poi)
    shp_2023_simulation = land_simulation_CA(shp_2017,2020,2029,caozhuang_station,path_road,path_road_osm,poi)
    shp_2023_simulation.to_file(r'2029_simluation.shp')









