# from tarfile import FIFOTYPE
import open3d as o3d
import numpy as np
import code,os,sys,struct    
import matplotlib.pyplot as plt
import time


### Visulization the original point clouds
def plot_raw_pcl(pcd):
    o3d.visualization.draw_geometries([pcd],
                                    width=1280,
                                    height=860,
                                    zoom=0.3,
                                    front=[ -0.028201869900951016, 0.22625896174693114, -0.97365884002729142 ],
                                    lookat=[ -2780.6908134776941, -590.79499333057709, 959.06654143345077 ],
                                    up=[ 0.042813262481875822, 0.97342585230251621, 0.22496474084793408 ])

### Generate bounding box and Pass through filter
class Filter3D_Passthrough():
    def __init__(self, is_rgb_4byte = False):
        self.is_rgb_4byte = is_rgb_4byte
        

    def rgb2float( self,r, g, b, a = 0 ):
        return struct.unpack('f', struct.pack('i',r << 16 | g << 8 | b))[0]

    def draw_guild_lines(self,dic, density = 0.5):
        new_col = []
        new_pos = []
        x_start,x_end = dic["x"]
        y_start,y_end = dic["y"]
        z_start,z_end = dic["z"]

        x_points,y_points,z_points = np.asarray(np.arange(x_start,x_end,density)),np.asarray(np.arange(y_start,y_end,density)),np.asarray(np.arange(z_start,z_end,density))
        
        y_starts,y_ends = np.asarray(np.full((len(x_points)),y_start)),np.asarray(np.full((len(x_points)),y_end))
        z_starts,z_ends = np.asarray(np.full((len(x_points)),z_start)),np.asarray(np.full((len(x_points)),z_end))
        lines_x = np.concatenate((np.vstack((x_points,y_starts,z_starts)).T,np.vstack((x_points,y_ends,z_starts)).T,np.vstack((x_points,y_starts,z_ends)).T,np.vstack((x_points,y_ends,z_ends)).T))


        x_starts,x_ends = np.asarray(np.full((len(y_points)),x_start)),np.asarray(np.full((len(y_points)),x_end))
        z_starts,z_ends = np.asarray(np.full((len(y_points)),z_start)),np.asarray(np.full((len(y_points)),z_end))
        lines_y = np.concatenate((np.vstack((x_starts,y_points,z_starts)).T,np.vstack((x_ends,y_points,z_starts)).T,np.vstack((x_starts,y_points,z_ends)).T,np.vstack((x_ends,y_points,z_ends)).T))


        x_starts,x_ends = np.asarray(np.full((len(z_points)),x_start)),np.asarray(np.full((len(z_points)),x_end))
        y_starts,y_ends = np.asarray(np.full((len(z_points)),y_start)),np.asarray(np.full((len(z_points)),y_end))
        lines_z = np.concatenate((np.vstack((x_starts,y_starts,z_points)).T,np.vstack((x_ends,y_starts,z_points)).T,np.vstack((x_starts,y_ends,z_points)).T,np.vstack((x_ends,y_ends,z_points)).T))

        if (self.is_rgb_4byte):
            lines_x_color =  np.full((len(lines_x)),self.rgb2float(255,0,0))#blue for x
            lines_y_color =  np.full((len(lines_y)),self.rgb2float(0,255,0))#green for y
            lines_z_color =  np.full((len(lines_z)),self.rgb2float(0,0,255))#red for z
            return np.concatenate((lines_x,lines_y,lines_z)),np.asmatrix(np.concatenate((lines_x_color,lines_y_color,lines_z_color))).T
        else:
            lines_x_color = np.zeros((len(lines_x),3))
            lines_y_color = np.zeros((len(lines_y),3))
            lines_z_color = np.zeros((len(lines_z),3))

            lines_x_color[:,0] = 1.0 #red for x
            lines_y_color[:,1] = 1.0 #green for y
            lines_z_color[:,2] = 1.0 #blue for z
            return np.concatenate((lines_x,lines_y,lines_z)),np.asmatrix(np.concatenate((lines_x_color,lines_y_color,lines_z_color)))
    
    def pass_through_filter(self, dic, pcd_fil):

        points = np.asarray(pcd_fil.points)
        colors = np.asarray(pcd_fil.colors)
        points[:,0]
        x_range = np.logical_and(points[:,0] >= dic["x"][0] ,points[:,0] <= dic["x"][1])
        y_range = np.logical_and(points[:,1] >= dic["y"][0] ,points[:,1] <= dic["y"][1])
        z_range = np.logical_and(points[:,2] >= dic["z"][0] ,points[:,2] <= dic["z"][1])

        pass_through_filter = np.logical_and(x_range,np.logical_and(y_range,z_range))

        pcd_fil.points = o3d.utility.Vector3dVector(points[pass_through_filter])
        pcd_fil.paint_uniform_color([0.25, 0.62, 1])
        # pcd.colors = o3d.utility.Vector3dVector(colors[pass_through_filter])

        return pcd_fil

### Generate bounding box of the pass through filter
def draw_bbox_passthrough(pcd):
    dic = {"x":[0,800],
            "y":[-400,1000],
            "z":[1000,2500]}
    #Drawing filter guidelines 
    new_pos, new_col = filPT.draw_guild_lines(dic)
    new_data = np.concatenate((new_pos, new_col),axis = 1)
    guild_points = o3d.geometry.PointCloud()
    guild_points.points = o3d.utility.Vector3dVector(new_pos)
    guild_points.colors = o3d.utility.Vector3dVector(new_col)

    #Pass through filtering
    print("Pass through filter")
    pcd_filtered = filPT.pass_through_filter(dic,pcd)

    return guild_points,pcd_filtered

### Prepare raw point cloud before filtering
def prepare_raw_pcl(pcd):
    # pcd = o3d.io.read_point_cloud(LandingPath)

    voxel_down_pcd = pcd.voxel_down_sample(voxel_size = 0.01)
    # voxel_down_pcd.paint_uniform_color([0.25, 0.62, 1])
    ### Visualization the raw point cloud
    o3d.visualization.draw_geometries([voxel_down_pcd],
                                    width=1280,
                                    height=860,
                                    zoom=0.74,
                                    front=[ -0.35237385107751878, 0.16213571919318584, -0.92170747943070697 ],
                                    lookat=[ -622.15540635158163, -106.14283325914268, 1558.9631819212902 ],
                                    up=[ 0.038047202656848432, 0.98654588887774564, 0.15899565877220476 ])
    return voxel_down_pcd

### Visulization the filter object
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    ### Visualization the filterd point cloud
    o3d.visualization.draw_geometries([inlier_cloud],
                                    width=1280,
                                    height=860,
                                    zoom=0.74,
                                    front=[ -0.35237385107751878, 0.16213571919318584, -0.92170747943070697 ],
                                    lookat=[ -622.15540635158163, -106.14283325914268, 1558.9631819212902 ],
                                    up=[ 0.038047202656848432, 0.98654588887774564, 0.15899565877220476 ])
    # o3d.io.write_point_cloud(FilteredPath,inlier_cloud)
    
    ### Visulization the eleminated point cloud
    outlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([outlier_cloud],
                                    width=1280,
                                    height=860,
                                    zoom=0.74,
                                    front=[ -0.35237385107751878, 0.16213571919318584, -0.92170747943070697 ],
                                    lookat=[ -622.15540635158163, -106.14283325914268, 1558.9631819212902 ],
                                    up=[ 0.038047202656848432, 0.98654588887774564, 0.15899565877220476 ])

    ### Visulization the both filterd and eleminated point cloud
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                    width=1280,
                                    height=860,
                                    zoom=0.74,
                                    front=[ -0.35237385107751878, 0.16213571919318584, -0.92170747943070697 ],
                                    lookat=[ -622.15540635158163, -106.14283325914268, 1558.9631819212902 ],
                                    up=[ 0.038047202656848432, 0.98654588887774564, 0.15899565877220476 ])

### Statiscal outlier removal (filtering)
def statistical_outlier_removal(voxel_down_pcd):
    print("Statistical outlier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=0.2)
    # display_inlier_outlier(voxel_down_pcd, ind)
    return voxel_down_pcd.select_by_index(ind)

### Radius outlier removal (filtering)
def radius_outlier_removal(voxel_down_pcd):
    print("Radius outlier removal")
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=8, radius=10)
    # display_inlier_outlier(voxel_down_pcd, ind)
    return voxel_down_pcd.select_by_index(ind)

### Density-based scan clustering (DBSCAN Clustering)
def dbscan_cluster(Filtered_PCL):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            Filtered_PCL.cluster_dbscan(eps=15, min_points=40, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    Filtered_PCL.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([Filtered_PCL],
    #                                     width=1280,
    #                                     height=860,
    #                                     zoom=0.52,
    #                                     front=[ 0.35841336406485025, 0.1045412555722367, -0.92769121281981404 ],
    #                                     lookat=[ -572.38828795693564, -374.20006974981737, 1582.1566409846967 ],
    #                                     up=[ -0.076723071324817052, 0.99364733736702315, 0.082331884892261439 ])
    return labels,Filtered_PCL

### Generate the bounding box of the container corner casting 
def draw_boundingbox_casting(cloud_pcd):
    dic = {"x":[0,0],
            "y":[0,0],
            "z":[0,0]}
    max_bound = o3d.geometry.OrientedBoundingBox.get_max_bound(cloud_pcd)
    min_bound = o3d.geometry.OrientedBoundingBox.get_min_bound(cloud_pcd)
    i = 0
    for eles in dic:
        dic[eles][0] = min_bound[i]
        dic[eles][1] = max_bound[i]
        i+=1

    new_pos, new_col = filPT.draw_guild_lines(dic)
    new_data = np.concatenate((new_pos, new_col),axis = 1)
    guild_points = o3d.geometry.PointCloud()
    guild_points.points = o3d.utility.Vector3dVector(new_pos)
    guild_points.colors = o3d.utility.Vector3dVector(new_col)
    return guild_points

### Generate bounding box of the twist lock
def draw_boundingbox_twl(cloud_pcd):
    dic = {"x":[0,0],
            "y":[0,0],
            "z":[0,0]}
    max_bound = o3d.geometry.OrientedBoundingBox.get_max_bound(cloud_pcd)
    min_bound = o3d.geometry.OrientedBoundingBox.get_min_bound(cloud_pcd)
    i = 0
    for eles in dic:
        dic[eles][0] = min_bound[i]
        dic[eles][1] = max_bound[i]
        i+=1

    new_pos, new_col = filPT.draw_guild_lines(dic)
    new_data = np.concatenate((new_pos, new_col),axis = 1)
    guild_points = o3d.geometry.PointCloud()
    guild_points.points = o3d.utility.Vector3dVector(new_pos)
    guild_points.colors = o3d.utility.Vector3dVector(new_col)
    return guild_points

### Select clusters from DBScan clustering
def select_cluster(labels_cluster,Filtered_PCL):
    unique, counts = np.unique(labels_cluster, return_counts=True)
    labels_dict = dict(zip(unique, counts))
    
    ### Choose point cloud of twist lock
    sorted_volume = np.argsort(counts)          ###Return the indices of the sorted value 
    sorted_volume = sorted_volume[::-1]         ###Reverse the order
    twistlock_pcd = Filtered_PCL
    for i in range(len(unique)):
        ind_twl = np.where(labels_cluster==sorted_volume[i]-1)[0]
        if sorted_volume[i]-1 != -1:            ###Eleminate the noise of the filtered point clouds
            temp_twl = Filtered_PCL.select_by_index(ind_twl)
            temp_maxbound = o3d.geometry.OrientedBoundingBox.get_max_bound(temp_twl)
            temp_minbound = o3d.geometry.OrientedBoundingBox.get_min_bound(temp_twl)
            
            ### Calculate the center coordinate of the twist lock's bounding box
            x_center = (temp_maxbound[0]+temp_minbound[0])/2
            y_center = (temp_maxbound[1]+temp_minbound[1])/2
            z_center = (temp_maxbound[2]+temp_minbound[2])/2

            if x_center <= 500 and x_center >= 300 and y_center <= -100 and y_center >= -350 and z_center <= 1900 and z_center >= 1500:
                twistlock_pcd = temp_twl
                break

    ### Draw bounding of the objects (Twist lock)
    bbox_twistlock = draw_boundingbox_twl(twistlock_pcd)
    return twistlock_pcd, bbox_twistlock

### Determine the top point of the twist lock
def detedt_toppoint(tw_pcd):
    y_threshold = o3d.geometry.OrientedBoundingBox.get_max_bound(tw_pcd)[1]-0.5       ##### Adjust the threshold to determine the top of the twist lock 
    points = np.array(tw_pcd.points)
    mask = points[:,1] > y_threshold
    dt_tw_pcd = o3d.geometry.PointCloud()
    outtop_tw_pcd = o3d.geometry.PointCloud()
    dt_tw_pcd.points = o3d.utility.Vector3dVector(points[mask])       #Keep the original color of point.

    dt_tw_pcd = tw_pcd.select_by_index(np.where(points[:,1] > y_threshold)[0])
    dt_tw_pcd.paint_uniform_color([1,0,0])
    outtop_tw_pcd = tw_pcd.select_by_index(np.where(points[:,1] <= y_threshold)[0])
    print(np.array(dt_tw_pcd.points)[0])
    return dt_tw_pcd,outtop_tw_pcd


def main():
    LandingPath = os.path.join(r'3D Camera\Dataset\process1\ply','20220101_103700_821_S1.ply')
    pcd = o3d.io.read_point_cloud(LandingPath)
    pcd.paint_uniform_color([0.25, 0.62, 1])

    ### Visualization the original point cloud
    # plot_raw_pcl(pcd)

    ### Generate the bounding box of the pass through filter and apply pass through filter
    bbox_passthrough,filtered_pthr_pcd = draw_bbox_passthrough(pcd)
    # pcd = o3d.io.read_point_cloud(LandingPath)
    # pcd.paint_uniform_color([0.25, 0.62, 1])
    # o3d.visualization.draw_geometries([bbox_passthrough,filtered_pthr_pcd],
    #                                 width=1280,
    #                                 height=860,
    #                                 zoom=0.3,
    #                                 front=[ -0.08411171125816469, -0.18314317675426217, -0.97948139177708748 ],
    #                                 lookat=[ -410.53803242958719, -201.5692266838862, 2166.5383972157665 ],
    #                                 up=[ 0.0070625310276934659, 0.98283043342841503, -0.18437586550955243 ])
    
    ### Prepare raw point cloud & Statistical outlier removal & Radius outlier removal
    # preprocess_pcl = prepare_raw_pcl(pcd)
    filtered_sta_out_remove = statistical_outlier_removal(filtered_pthr_pcd)
    Filtered_PCL = radius_outlier_removal(filtered_sta_out_remove) 
    # o3d.visualization.draw_geometries([Filtered_PCL])

    ### DBScan clustering
    labels_cluster,dbs_fil_pcl = dbscan_cluster(Filtered_PCL)
    # o3d.visualization.draw_geometries([dbs_fil_pcl])


    ### Select cluster from the DBScan results
    twistlock_pcd, bbox_twistlock = select_cluster(labels_cluster,dbs_fil_pcl)
    
    ### Determine the top point cloud of the twistlock
    dt_tw_pcd,outtop_tw_pcd = detedt_toppoint(twistlock_pcd)

    ### Calculate the execution time
    print(time.time() - start_time, "seconds")

    ### Visualization the result
    # pcd = o3d.io.read_point_cloud(LandingPath)
    # pcd.paint_uniform_color([0.25, 0.62, 1])
    # print(o3d.geometry.OrientedBoundingBox.get_max_bound(dt_tw_pcd))
    outtop_tw_pcd.paint_uniform_color([0, 0.5, 0.75])
    o3d.visualization.draw_geometries([dt_tw_pcd,outtop_tw_pcd],
                                    width=1280,
                                    height=860,
                                    zoom=2.0,
                                    front=[ -0.18605305458219931, 0.30777213928635117, -0.93309194142899776 ],
                                    lookat=[ 428.57040405273438, -231.31217956542969, 1747.9197998046875 ],
                                    up=[ 0.064737140833148937, 0.95144983295990382, 0.30091912195359116 ])

if __name__ == "__main__":
    filPT = Filter3D_Passthrough()
    start_time = time.time()
    main()