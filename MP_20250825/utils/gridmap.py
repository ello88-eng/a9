from functools import total_ordering

import matplotlib.pyplot as plt
import numpy as np
from pymap3d.enu import geodetic2enu
from scipy.spatial.transform import Rotation


def rotate_matrix_2d(angle: float) -> np.ndarray:
    return Rotation.from_euler(seq="z", angles=angle).as_matrix()[0:2, 0:2]


@total_ordering
class FloatGrid:

    def __init__(self, init_val=0.0):
        self.data = init_val

    def get_float_data(self):
        return self.data

    def __eq__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() == other.get_float_data()

    def __lt__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() < other.get_float_data()


class GridMap:
    """
    GridMap class
    """

    def __init__(
        self,
        width: int = 200,
        height: int = 200,
        resolution: int = 50,
        center_x: float = 0.0,
        center_y: float = 0.0,
        init_val=FloatGrid(0.0),
    ):
        """__init__

        :param width: number of grid for width
        :param height: number of grid for height
        :param resolution: grid resolution [m]
        :param center_x: center x position  [m]
        :param center_y: center y position [m]
        :param init_val: initial value for all grid
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.center_x = center_x
        self.center_y = center_y
        self.left_lower_x = self.center_x - self.width / 2.0 * self.resolution
        self.left_lower_y = self.center_y - self.height / 2.0 * self.resolution
        self.n_data = self.width * self.height
        self.data = [init_val] * self.n_data
        self.data_type = type(init_val)

    def get_value_from_xy_index(self, x_ind, y_ind):
        """get_value_from_xy_index

        when the index is out of grid map area, return None

        :param x_ind: x index
        :param y_ind: y index
        """

        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)

        if 0 <= grid_ind < self.n_data:
            return self.data[grid_ind]
        else:
            return None

    def get_xy_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(y_pos, self.left_lower_y, self.height)

        return x_ind, y_ind

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)

        if (not x_ind) or (not y_ind):
            return False  # NG

        flag = self.set_value_from_xy_index(x_ind, y_ind, val)

        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (x_ind is None) or (y_ind is None):
            return False, False

        grid_ind = int(y_ind * self.width + x_ind)

        if 0 <= grid_ind < self.n_data and isinstance(val, self.data_type):
            self.data[grid_ind] = val
            return True  # OK
        else:
            return False  # NG

    def set_value_from_polygon(self, pol_x, pol_y, val, inside=True):
        """set_value_from_polygon

        Setting value inside or outside polygon

        :param pol_x: x position list for a polygon
        :param pol_y: y position list for a polygon
        :param val: grid value
        :param inside: setting data inside or outside
        """

        # making ring polygon
        if (pol_x[0] != pol_x[-1]) or (pol_y[0] != pol_y[-1]):
            np.append(pol_x, pol_x[0])
            np.append(pol_y, pol_y[0])

        # setting value for all grid
        for x_ind in range(self.width):
            for y_ind in range(self.height):
                x_pos, y_pos = self.calc_grid_central_xy_position_from_xy_index(
                    x_ind, y_ind
                )

                flag = self.check_inside_polygon(x_pos, y_pos, pol_x, pol_y)

                if flag is inside:
                    self.set_value_from_xy_index(x_ind, y_ind, val)

    def set_value_from_footprints(
        self, footprints: np.ndarray, value: float, inside: bool = True
    ) -> None:
        enu_footprints = []
        for footprint in footprints:
            for fp in footprint:
                enu_corners = geodetic2enu(
                    lat=fp[0], lon=fp[1], h=0, lat0=0, lon0=0, h0=0
                )[:2]
                enu_footprints.append(list(enu_corners))

        xf = [footprint[0] for footprint in enu_footprints]
        yf = [footprint[1] for footprint in enu_footprints]

        enu_footprints = np.stack(
            [np.array(xf).reshape(-1, 4), np.array(yf).reshape(-1, 4)], axis=1
        )
        first_element = enu_footprints[:, :, 0, np.newaxis]
        enu_footprints = np.concatenate(
            (enu_footprints, first_element), axis=2
        ).tolist()

        for enu_footprint in enu_footprints:
            for x_index in range(self.num_width):
                for y_index in range(self.num_height):
                    x_position, y_position = (
                        self.calc_grid_central_xy_position_from_xy_index(
                            x_index, y_index
                        )
                    )
                    flag = self.check_inside_polygon(
                        x_position, y_position, enu_footprint[0], enu_footprint[1]
                    )
                    is_outside = self.check_occupied_from_xy_index(
                        x_index, y_index, 1.0
                    )
                    if flag is inside and not is_outside:
                        self.set_value_from_xy_index(x_index, y_index, value)

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        grid_ind = int(y_ind * self.width + x_ind)
        return grid_ind

    def calc_xy_index_from_grid_index(self, grid_ind):
        y_ind, x_ind = divmod(grid_ind, self.width)
        return x_ind, y_ind

    def calc_grid_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(y_pos, self.left_lower_y, self.height)

        return self.calc_grid_index_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_grid_index(self, grid_ind):
        x_ind, y_ind = self.calc_xy_index_from_grid_index(grid_ind)
        return self.calc_grid_central_xy_position_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        x_pos = self.calc_grid_central_xy_position_from_index(x_ind, self.left_lower_x)
        y_pos = self.calc_grid_central_xy_position_from_index(y_ind, self.left_lower_y)

        return x_pos, y_pos

    def calc_grid_central_xy_position_from_index(self, index, lower_pos):
        return lower_pos + index * self.resolution + self.resolution / 2.0

    def calc_xy_index_from_position(self, pos, lower_pos, max_index):
        ind = int(np.floor((pos - lower_pos) / self.resolution))
        if 0 <= ind <= max_index:
            return ind
        else:
            return None

    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val):

        val = self.get_value_from_xy_index(x_ind, y_ind)

        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def expand_grid(self, occupied_val=FloatGrid(1.0)):
        x_inds, y_inds, values = [], [], []

        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy, occupied_val):
                    x_inds.append(ix)
                    y_inds.append(iy)
                    values.append(self.get_value_from_xy_index(ix, iy))

        for ix, iy, value in zip(x_inds, y_inds, values):
            self.set_value_from_xy_index(ix + 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy + 1, val=value)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy - 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=value)

    @staticmethod
    def check_inside_polygon(iox, ioy, x, y):

        n_point = len(x) - 1
        inside = False
        for i1 in range(n_point):
            i2 = (i1 + 1) % (n_point + 1)

            if x[i1] >= x[i2]:
                min_x, max_x = x[i2], x[i1]
            else:
                min_x, max_x = x[i1], x[i2]
            if not min_x <= iox < max_x:
                continue

            tmp1 = (y[i2] - y[i1]) / (x[i2] - x[i1])
            if (y[i1] + tmp1 * (iox - x[i1]) - ioy) > 0.0:
                inside = not inside

        return inside

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)
        print("n_data:", self.n_data)

    def plot_grid_map(self, ax=None):
        float_data_array = np.array([d.get_float_data() for d in self.data])
        grid_data = np.reshape(float_data_array, (self.height, self.width))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.axis("equal")

        return heat_map


def polygon_set_demo():
    ox = [-2345.24, 2273.8100000000004, 2273.8100000000004, -2345.24, -2345.24]
    oy = [392.857, 392.857, -4678.573, -4678.573, 392.857]
    roi_area = calculate_corners(-2345.24, 392.857, 4619.05, 5071.43)
    grid_map = GridMap(
        200,
        200,
        resolution=int(np.linalg.norm(roi_area[0] - roi_area[1]) / 200),
        center_x=np.mean([roi_area[0], roi_area[2]], axis=0)[0],
        center_y=np.mean([roi_area[0], roi_area[2]], axis=0)[1],
    )
    grid_map.set_value_from_polygon(ox, oy, FloatGrid(1.0), inside=False)
    grid_map.plot_grid_map()
    plt.axis("equal")
    plt.grid(True)


def rot_mat_2d(angle: float) -> np.ndarray:
    """Create 2D rotation matrix from an angle.

    Args:
        angle (float): _description_

    Returns:
        np.ndarray: A 2D rotation matrix.
    """
    return Rotation.from_euler(seq="z", angles=angle).as_matrix()[0:2, 0:2]


def calculate_corners(x: float, y: float, w: float, h: float):
    top_left = np.array([x, y])
    top_right = np.array([x + w, y])
    bottom_right = np.array([x + w, y - h])
    bottom_left = np.array([x, y - h])

    return top_left, top_right, bottom_right, bottom_left


def position_set_demo():
    roi_area = calculate_corners(-2345.24, 392.857, 4619.05, 5071.43)
    print(roi_area)
    grid_map = GridMap(
        200,
        200,
        resolution=int(np.linalg.norm(roi_area[0] - roi_area[1]) / 200),
        center_x=np.mean([roi_area[0], roi_area[2]], axis=0)[0],
        center_y=np.mean([roi_area[0], roi_area[2]], axis=0)[1],
    )
    grid_map.plot_grid_map()
    plt.axis("equal")
    plt.grid(True)


def main():
    position_set_demo()
    polygon_set_demo()
    plt.show()
