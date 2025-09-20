import numpy as np
import os
import open3d


def getDir(path):
    fList = os.listdir(path)
    F = []
    for i in fList:
        d = os.path.join(path,i)
        if os.path.isdir(d):
            F.append(i)
    return F


def getFilenames(path):
    Dir = os.path.join(path)
    fList = os.listdir(Dir)
    files = []
    for i in fList:
        if i[-4:] == ".npy":
            f = os.path.join(Dir, i)
            if os.path.isfile(f):
                files.append(i)
    return files


def scale_to_unit_cube(x):
    if len(x) == 0:
        return x
    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x


def readFile(path, fileName):
    path = os.path.join(path, fileName)
    pc = np.load(path)
    return pc


def saveFile(data, path, fileName):
    path = os.path.join(path, fileName)
    np.save(path, data)


def compute_norm_and_curvature(pc, knn_indices=None):
    if knn_indices is not None:
        pc = pc[knn_indices]
    covariance = np.cov(pc.T)
    w, v = np.linalg.eig(covariance)
    v = v.T
    w = np.real(w)
    i = np.argmin(np.abs(w))
    norm = v[i]
    curv = w[i] / np.sum(np.abs(w))
    # assert curv is not complex
    return norm, np.real(curv)


def compute_norm_included_angle(q_norm, norms, knn_indices=None):
    norm_included_angles = []
    norm_included_angle = 0
    if knn_indices is not None:
        for i in range(len(knn_indices)):
            cosine_value = np.dot(q_norm, norms[i]) / (np.linalg.norm(q_norm) * (np.linalg.norm(norms[i])))
            eps = 1e-6
            if 1.0 < cosine_value < 1.0 + eps:
                cosine_value = 1.0
            elif -1.0 - eps < cosine_value < -1.0:
                cosine_value = -1.0
            included_angle = np.arccos(cosine_value) * 180 / np.pi
            norm_included_angles.append(included_angle)
        norm_included_angle = np.average(norm_included_angles)
    return norm_included_angle


if __name__ == '__main__':
    root = "PointSegDAdataset/scape/train/"
    save_root = "PointSegDAdataset/scape/train/"
    PC = []
    Curv = []
    Label = []
    File = []
    files = getFilenames(root)
    files.sort() 
    for f in files:
        print(f)
        File.append(f)
        data = readFile(root, f)  # (2048, 3)
        pc = data[:, :3]
        label = data[:,3]
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(pc)  # open3d.Vector3dVector()
        kdtree = open3d.geometry.KDTreeFlann()
        kdtree.set_geometry(point_cloud)
        norms = []
        curvs = []
        norm_included_angles = []
        for j in range(pc.shape[0]):
            q = pc[j]
            q = np.float64(q)
            k, indices, dist = kdtree.search_knn_vector_3d(q, knn=32)
            indices = np.asarray(indices)
            _, curv = compute_norm_and_curvature(pc, indices)
            curvs.append(curv)
        curvs = np.array(curvs).reshape(pc.shape[0], 1)
        print("PC:", pc.shape)
        print("curvs", curvs.shape)
        PC.append(pc)
        Curv.append(curvs)
        Label.append(label)

    assert (len(PC)==len(Curv))
    print(len(PC))
    print(len(Curv))
    print(len(File))

    Curv = np.array(Curv)
    cmin = np.min(Curv)
    cmax = np.max(Curv)
    Curv = 2*(Curv-cmin)/(cmax-cmin)-1
    for j in range(len(PC)):
        f = File[j]
        pc = PC[j]
        curv = Curv[j].reshape(pc.shape[0], 1)  # note that the point number of pointcloud in shapenet is 1024
        label = Label[j].reshape(pc.shape[0], 1)
        data = np.concatenate([pc, label, curv], -1)
        # save_path = os.path.join(save_root, str(i))
        if not os.path.exists(os.path.join(save_root)):
            os.makedirs(os.path.join(save_root))
        saveFile(data, save_root, f)









