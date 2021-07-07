import numpy as np

class potential_field_interpolation:
    def __init__(self, df_int, df_or, C0, a):
        self.df_int = pick_ref_points(df_int)
        self.df_or = df_or
        self.C0 = C0
        self.a = a
        self.formations = list(set(self.df_int['formation'].to_list()))

    def covariance_function(self, r):
        # r: distance between points
        r = np.abs(r)  # r is always positive except when C_r_prime and C_r_doubleprime are calculated at r=0
        r_over_a = r / self.a
        C_r = (r >= 0) * (r <= self.a) * self.C0 * (
                1 - 7 * r_over_a ** 2 + 35 / 4 * r_over_a ** 3 - 7 / 2 * r_over_a ** 5 + 3 / 4 * r_over_a ** 7)
        return C_r

    def covariance_function_prime(self, r):
        c_prime_over_r = (r >= 0) * (r <= self.a) * self.C0 * (-14 / self.a ** 2 + 105 / 4 / self.a ** 3 * r -
                                                        35 / 2 / self.a ** 5 * r ** 3 + 21 / 4 / self.a ** 7 * r ** 5)  # c_prime / r

        return c_prime_over_r

    def covariance_function_doubleprime(self, r):
        c_doubleprime = (r >= 0) * (r <= self.a) * self.C0 * (-14 / self.a ** 2 + 210 / 4 / self.a ** 3 * r -
                                                              140 / 2 / self.a ** 5 * r ** 3 + 126 / 4 / self.a ** 7 * r ** 5)
        return c_doubleprime

    def covariance_function_Z2u_Z2u(self, r, hu):
        # hu: distance between points along 'u' axis
        Cr_prime_over_r = self.covariance_function_prime(r)
        Cr_doubleprime = self.covariance_function_doubleprime(r)
        if r == 0:
            C_Z2u_Z2u = -Cr_prime_over_r
        else:
            factor = hu ** 2 / r ** 2
            C_Z2u_Z2u = factor * Cr_prime_over_r - factor * Cr_doubleprime - Cr_prime_over_r

        return C_Z2u_Z2u

    def covariance_function_Z2u_Z2v(self, r, hu, hv):
        # hu: distance between points along 'u' axis
        # hv: distance between points along 'v' axis
        Cr_prime_over_r = self.covariance_function_prime(r)
        Cr_doubleprime = self.covariance_function_doubleprime(r)
        if r == 0:
            C_Z2u_Z2v = 0
        else:
            factor = (hu * hv / r ** 2)
            C_Z2u_Z2v = factor * Cr_prime_over_r - factor * Cr_doubleprime
        return C_Z2u_Z2v

    def covariance_function_Z1new_Z2(self, r, hu):
        # hu: distance between points along 'u' axis
        Cr_prime_over_r = self.covariance_function_prime(r)
        C_Z1new_Z2 = -hu * Cr_prime_over_r
        return C_Z1new_Z2

    def covariance_Z1new_Z1new(self):
        # df_int: data frame of the interfaces locations
        ncov = size_covariance_Z_Z(self.df_int)
        C_Z1new_Z1new = np.zeros((ncov, ncov))
        ii = 0
        for layeri in self.formations:
            df_refi = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(self.df_int.point_type).get_group('ref')
            df_resti = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(self.df_int.point_type).get_group('rest')
            x_refi = make_array_from_df(df_refi)
            x_resti = make_array_from_df(df_resti)
            n_resti = len(df_resti)
            for irest_i in range(n_resti):
                jj = 0
                for layerj in self.formations:
                    df_refj = self.df_int.groupby(self.df_int.formation).get_group(layerj).groupby(self.df_int.point_type).get_group('ref')
                    df_restj = self.df_int.groupby(self.df_int.formation).get_group(layerj).groupby(self.df_int.point_type).get_group('rest')
                    x_refj = make_array_from_df(df_refj)
                    x_restj = make_array_from_df(df_restj)
                    n_restj = len(df_restj)
                    for jrest_j in range(n_restj):
                        ra = np.linalg.norm(x_resti[irest_i, :] - x_restj[jrest_j, :])
                        rb = np.linalg.norm(x_refi - x_restj[jrest_j, :])
                        rc = np.linalg.norm(x_resti[irest_i, :] - x_refj)
                        rd = np.linalg.norm(x_refi - x_refj)

                        C_resti_restj = self.covariance_function(ra)
                        C_refi_restj = self.covariance_function(rb)
                        C_resti_refj = self.covariance_function(rc)
                        C_refi_refj = self.covariance_function(rd)

                        C_Z1new_Z1new[ii, jj] = C_resti_restj - C_refi_restj - C_resti_refj + C_refi_refj
                        jj = jj + 1
                ii = ii + 1

        self.C_Z1new_Z1new = C_Z1new_Z1new

    def covariance_Z2_Z2(self):
        n_or = len(self.df_or)
        x_beta = make_array_from_df(self.df_or)
        C_Z2_Z2 = np.zeros((n_or * 3, n_or * 3))

        ii = 0
        for ixyz in range(3):
            for ior in range(n_or):
                jj = 0
                for jxyz in range(3):
                    for jor in range(n_or):
                        r = np.linalg.norm(x_beta[ior, :] - x_beta[jor, :])
                        hu = x_beta[ior, ixyz] - x_beta[jor, ixyz]
                        hv = x_beta[ior, jxyz] - x_beta[jor, jxyz]
                        if ixyz == jxyz:
                            C_Z2_Z2[ii, jj] = self.covariance_function_Z2u_Z2u(r, hu)
                        else:
                            C_Z2_Z2[ii, jj] = self.covariance_function_Z2u_Z2v(r, hu, hv)

                        jj = jj + 1

                ii = ii + 1

        self.C_Z2_Z2 = C_Z2_Z2

    def covariance_Z1new_Z2(self):
        x_beta = make_array_from_df(self.df_or)
        ncov_cols = np.shape(self.C_Z2_Z2)[1]
        ncov_rows = np.shape(self.C_Z1new_Z1new)[0]
        C_Z1new_Z2 = np.zeros((ncov_rows, ncov_cols))

        ii = 0
        for layeri in self.formations:
            df_refi = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(self.df_int.point_type).get_group('ref')
            df_resti = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(self.df_int.point_type).get_group('rest')
            x_refi = make_array_from_df(df_refi)
            x_resti = make_array_from_df(df_resti)
            n_resti = len(df_resti)
            for irest_i in range(n_resti):
                jj = 0
                for ixyz in range(3):
                    for ibeta in range(len(x_beta)):
                        ra = np.linalg.norm(x_resti[irest_i, :] - x_beta[ibeta, :])
                        rb = np.linalg.norm(x_refi - x_beta[ibeta, :])
                        hu_a = x_resti[irest_i, ixyz] - x_beta[ibeta, ixyz]
                        hu_b = x_refi[0, ixyz] - x_beta[ibeta, ixyz]

                        C_resti_beta = self.covariance_function_Z1new_Z2(ra, hu_a)
                        C_refi_beta = self.covariance_function_Z1new_Z2(rb, hu_b)
                        C_Z1new_Z2[ii, jj] = C_resti_beta - C_refi_beta

                        jj = jj + 1
                ii = ii + 1

        self.C_Z1new_Z2 = C_Z1new_Z2

    def drift_U_Z1new(self):
        nrest = len(self.df_int) - len(self.formations)
        if self.drift_type == 'first_order':
            U_Z1new = np.zeros((nrest, 3))
            ii = 0
            for layeri in self.formations:
                df_refi = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(
                    self.df_int.point_type).get_group('ref')
                df_resti = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(
                    self.df_int.point_type).get_group('rest')
                x_refi = np.squeeze(make_array_from_df(df_refi))
                x_resti = make_array_from_df(df_resti)
                n_resti = len(df_resti)
                for irest_i in range(n_resti):
                    U_Z1new[ii, 0] = x_resti[irest_i, 0] - x_refi[0]  # x - x0
                    U_Z1new[ii, 1] = x_resti[irest_i, 1] - x_refi[1]  # y - y0
                    U_Z1new[ii, 2] = x_resti[irest_i, 2] - x_refi[2]  # z - z0

                    ii = ii + 1

        if self.drift_type == 'second_order':
            U_Z1new = np.zeros((nrest, 9))
            ii = 0
            for layeri in self.formations:
                df_refi = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(
                    self.df_int.point_type).get_group('ref')
                df_resti = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(
                    self.df_int.point_type).get_group('rest')
                x_refi = np.squeeze(make_array_from_df(df_refi))
                x_resti = make_array_from_df(df_resti)
                n_resti = len(df_resti)
                for irest_i in range(n_resti):
                    U_Z1new[ii, 0] = x_resti[irest_i, 0] - x_refi[0]  # x - x0
                    U_Z1new[ii, 1] = x_resti[irest_i, 1] - x_refi[1]  # y - y0
                    U_Z1new[ii, 2] = x_resti[irest_i, 2] - x_refi[2]  # z - z0
                    U_Z1new[ii, 3] = x_resti[irest_i, 0] ** 2 - x_refi[0] ** 2  # x ** 2 - x0 ** 2
                    U_Z1new[ii, 4] = x_resti[irest_i, 1] ** 2 - x_refi[1] ** 2  # y ** 2 - y0 ** 2
                    U_Z1new[ii, 5] = x_resti[irest_i, 2] ** 2 - x_refi[2] ** 2  # z ** 2 - z0 ** 2
                    U_Z1new[ii, 6] = x_resti[irest_i, 0] * x_resti[irest_i, 1] - x_refi[0] * x_refi[1]  # xy - x0y0
                    U_Z1new[ii, 7] = x_resti[irest_i, 0] * x_resti[irest_i, 2] - x_refi[0] * x_refi[2]  # xz - x0z0
                    U_Z1new[ii, 8] = x_resti[irest_i, 1] * x_resti[irest_i, 2] - x_refi[1] * x_refi[2]  # yz - y0z0
                    ii = ii + 1
        self.U_Z1new = U_Z1new

    def drift_U_Z2(self):
        nori = len(self.df_or)
        x_beta = make_array_from_df(self.df_or)
        if self.drift_type == 'first_order':
            U_Z2 = np.zeros((nori * 3, 3))
            for ior in range(nori):
                U_Z2[0*nori+ior, 0] = 1  # dxbeta / dx
                U_Z2[0*nori+ior, 1] = 0  # dybeta / dx
                U_Z2[0*nori+ior, 2] = 0  # dzbeta / dx

                U_Z2[1*nori+ior, 0] = 0  # dxbeta / dy
                U_Z2[1*nori+ior, 1] = 1  # dybeta / dy
                U_Z2[1*nori+ior, 2] = 0  # dzbeta / dy

                U_Z2[2*nori+ior, 0] = 0  # dxbeta / dz
                U_Z2[2*nori+ior, 1] = 0  # dybeta / dz
                U_Z2[2*nori+ior, 2] = 1  # dzbeta / dz

        if self.drift_type == 'second_order':
            U_Z2 = np.zeros((nori * 3, 9))
            for ior in range(nori):
                U_Z2[0*nori+ior, 0] = 1  # dxbeta / dx
                U_Z2[0*nori+ior, 1] = 0  # dybeta / dx
                U_Z2[0*nori+ior, 2] = 0  # dzbeta / dx
                U_Z2[0*nori+ior, 3] = 2 * x_beta[ior, 0]  # dxbeta**2 / dx
                U_Z2[0*nori+ior, 4] = 0  # dybeta**2 / dx
                U_Z2[0*nori+ior, 5] = 0  # dzbeta**2 / dx
                U_Z2[0*nori+ior, 6] = x_beta[ior, 1]  # dxbetaybeta / dx
                U_Z2[0*nori+ior, 7] = x_beta[ior, 2]  # dxbetazbeta / dx
                U_Z2[0*nori+ior, 8] = 0  # dybetazbeta / dx

                U_Z2[1*nori+ior, 0] = 0  # dxbeta / dy
                U_Z2[1*nori+ior, 1] = 1  # dybeta / dy
                U_Z2[1*nori+ior, 2] = 0  # dzbeta / dy
                U_Z2[1*nori+ior, 3] = 0  # dxbeta**2 / dy
                U_Z2[1*nori+ior, 4] = 2 * x_beta[ior, 1]  # dybeta**2 / dy
                U_Z2[1*nori+ior, 5] = 0  # dzbeta**2 / dy
                U_Z2[1*nori+ior, 6] = x_beta[ior, 0]  # dxbetaybeta / dy
                U_Z2[1*nori+ior, 7] = 0  # dxbetazbeta / dy
                U_Z2[1*nori+ior, 8] = x_beta[ior, 2]  # dybetazbeta / dy

                U_Z2[2*nori+ior, 0] = 0  # dxbeta / dz
                U_Z2[2*nori+ior, 1] = 0  # dybeta / dz
                U_Z2[2*nori+ior, 2] = 1  # dzbeta / dz
                U_Z2[2*nori+ior, 3] = 0  # dxbeta**2 / dz
                U_Z2[2*nori+ior, 4] = 0  # dybeta**2 / dz
                U_Z2[2*nori+ior, 5] = 2 * x_beta[ior, 2]  # dzbeta**2 / dz
                U_Z2[2*nori+ior, 6] = 0  # dxbetaybeta / dz
                U_Z2[2*nori+ior, 7] = x_beta[ior, 0]  # dxbetazbeta / dz
                U_Z2[2*nori+ior, 8] = x_beta[ior, 1]  # dybetazbeta / dz

        self.U_Z2 = U_Z2

    def assemble_covariance(self, drift_type='first_order'):
        self.drift_type = drift_type
        if drift_type == 'zero_order':
            C1 = np.hstack((self.C_Z1new_Z1new, self.C_Z1new_Z2))
            C2 = np.hstack((self.C_Z1new_Z2.transpose(), self.C_Z2_Z2))
            C_cokriging = np.vstack((C1, C2))

        else:  # first and second order drift
            self.drift_U_Z1new()
            self.drift_U_Z2()

            C1 = np.hstack((self.C_Z1new_Z1new, self.C_Z1new_Z2, self.U_Z1new))
            C2 = np.hstack((self.C_Z1new_Z2.transpose(), self.C_Z2_Z2, self.U_Z2))
            C3 = np.hstack((self.U_Z1new.transpose(), self.U_Z2.transpose(), np.zeros((np.shape(self.U_Z1new)[1], np.shape(self.U_Z1new)[1]))))

            C_cokriging = np.vstack((C1, C2, C3))

        self.C_cokriging = C_cokriging

    def add_nugget(self):
        nugget = 10 ** -12
        self.C_cokriging = self.C_cokriging + nugget * np.eye(np.shape(self.C_cokriging)[0], np.shape(self.C_cokriging)[1])

    def invert_covariance(self):
        try:
            self.C_inv = np.linalg.inv(self.C_cokriging)
            print('No nugget required')
        except:
            self.add_nugget()
            self.C_inv = np.linalg.inv(self.C_cokriging)
            print('Nugget added')

    def make_grid(self, x_ini, x_end, z_ini, z_end, dl=0.5):
        x = np.arange(x_ini, x_end + dl, dl)
        z = np.arange(z_ini, z_end + dl, dl)
        self.X_grid, self.Z_grid = np.meshgrid(x, z)

    def covariance_Z1new_vector(self):
        nx = np.shape(self.X_grid)[1]
        nz = np.shape(self.Z_grid)[0]
        ncov = size_covariance_Z_Z(self.df_int)
        covZ1new_vector = np.zeros((ncov, nx * nz))

        ii = 0
        for layeri in self.formations:
            df_refi = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(
                self.df_int.point_type).get_group('ref')
            df_resti = self.df_int.groupby(self.df_int.formation).get_group(layeri).groupby(
                self.df_int.point_type).get_group('rest')
            x_refi = make_array_from_df(df_refi)
            x_resti = make_array_from_df(df_resti)
            n_resti = len(df_resti)
            for irest_i in range(n_resti):
                jj = 0
                for iz in range(nz):
                    for ix in range(nx):
                        point_i = np.array([self.X_grid[iz, ix], 0, self.Z_grid[iz, ix]])
                        ra = np.linalg.norm(x_resti[irest_i, :] - point_i)
                        rb = np.linalg.norm(x_refi - point_i)
                        C_resti_point = self.covariance_function(ra)
                        C_refi_point = self.covariance_function(rb)
                        covZ1new_vector[ii, jj] = C_resti_point - C_refi_point
                        jj = jj + 1
                ii = ii + 1

        self.covZ1new_vector = covZ1new_vector

    def covariance_Z1new_Z2vector(self):
        nx = np.shape(self.X_grid)[1]
        nz = np.shape(self.Z_grid)[0]
        ncov = np.shape(self.C_Z2_Z2)[1]
        covZ1new_Z2_vector = np.zeros((ncov, nx * nz))

        x_beta = make_array_from_df(self.df_or)

        for ixyz in range(3):
            ii = ixyz * len(x_beta)
            for ibeta in range(len(x_beta)):
                jj = 0
                for iz in range(nz):
                    for ix in range(nx):
                        point_i = np.array([self.X_grid[iz, ix], 0, self.Z_grid[iz, ix]])

                        ra = np.linalg.norm(point_i - x_beta[ibeta, :])
                        hu_a = -x_beta[ibeta, ixyz] + point_i[ixyz]

                        C_resti_beta = self.covariance_function_Z1new_Z2(ra, hu_a)

                        covZ1new_Z2_vector[ii, jj] = C_resti_beta
                        jj = jj + 1

                ii = ii + 1

        self.covZ1new_Z2_vector = covZ1new_Z2_vector

    def drift_vector(self):
        nx = np.shape(self.X_grid)[1]
        nz = np.shape(self.Z_grid)[0]
        if self.drift_type == 'zero_order':
            pass
        elif self.drift_type == 'first_order':
            ndrift = 3
            f_Z1new = np.zeros((ndrift, nx * nz))
            jj = 0
            for iz in range(nz):
                for ix in range(nx):
                    point_i = np.array([self.X_grid[iz, ix], 0, self.Z_grid[iz, ix]])
                    f_Z1new[0, jj] = point_i[0]
                    f_Z1new[1, jj] = point_i[1]
                    f_Z1new[2, jj] = point_i[2]

                    jj = jj + 1

            self.f_Z1new = f_Z1new
        else:
            ndrift = 9
            f_Z1new = np.zeros((ndrift, nx * nz))
            jj = 0
            for iz in range(nz):
                for ix in range(nx):
                    point_i = np.array([self.X_grid[iz, ix], 0, self.Z_grid[iz, ix]])
                    f_Z1new[0, jj] = point_i[0]
                    f_Z1new[1, jj] = point_i[1]
                    f_Z1new[2, jj] = point_i[2]
                    f_Z1new[3, jj] = point_i[0] ** 2
                    f_Z1new[4, jj] = point_i[1] ** 2
                    f_Z1new[5, jj] = point_i[2] ** 2
                    f_Z1new[6, jj] = point_i[0] * point_i[1]
                    f_Z1new[7, jj] = point_i[0] * point_i[2]
                    f_Z1new[8, jj] = point_i[1] * point_i[2]

                    jj = jj + 1

            self.f_Z1new = f_Z1new

    def calc_potential_field(self):
        nx = np.shape(self.X_grid)[1]
        nz = np.shape(self.Z_grid)[0]

        ncov = size_covariance_Z_Z(self.df_int)

        self.invert_covariance()

        self.covariance_Z1new_vector()
        self.covariance_Z1new_Z2vector()
        self.drift_vector()

        if self.drift_type == 'zero_order':
            right_vector = np.vstack((self.covZ1new_vector, self.covZ1new_Z2_vector))
        elif self.drift_type == 'first_order':
            right_vector = np.vstack((self.covZ1new_vector, self.covZ1new_Z2_vector, self.f_Z1new))
        else:
            right_vector = np.vstack((self.covZ1new_vector, self.covZ1new_Z2_vector, self.f_Z1new))

        weights = np.matmul(self.C_inv, right_vector)
        weights_Z2 = weights[ncov:ncov+np.shape(self.covZ1new_Z2_vector)[0], :].transpose()

        Z2 = make_orientation_vector(self.df_or)

        Z1_est = np.matmul(weights_Z2, Z2).reshape((nz, nx))
        self.Z1_est = Z1_est


# Useful functions
def pick_ref_points(df_int):
    formations = list(set(df_int['formation'].to_list()))
    ref_list = len(df_int) * ['rest']
    for iform, formi in enumerate(formations):
        for idf, formdfi in enumerate(df_int['formation']):
            if formdfi == formi:
                ref_list[idf] = 'ref'
                break

    df_int['point_type'] = ref_list
    return df_int


def make_array_from_df(df):
    n = len(df)
    x = np.zeros((n, 3))
    x[:, 0] = df['X']
    x[:, 1] = df['Y']
    x[:, 2] = df['Z']
    return x


def size_covariance_Z_Z(df_int):
    formations = list(set(df_int['formation'].to_list()))
    ncov = 0
    for layeri in formations:
        df_resti = df_int.groupby(df_int.formation).get_group(layeri).groupby(df_int.point_type).get_group('rest')
        ncov = ncov + len(df_resti)

    return ncov


def make_orientation_vector(df_or):
    n_or = len(df_or)
    orientation_vector = np.zeros((n_or * 3, 1))
    count = 0
    for i in range(3):
        for j in range(n_or):
            dip_i = df_or['dip'][j]
            az_i = df_or['azimuth'][j]
            if i == 0:
                orientation_vector[count, 0] = np.sin(dip_i * np.pi / 180) * np.sin(az_i * np.pi / 180)
            if i == 1:
                orientation_vector[count, 0] = 0
            if i == 2:
                orientation_vector[count, 0] = np.cos(dip_i * np.pi / 180)

            count = count + 1

    return orientation_vector









