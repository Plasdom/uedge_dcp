import numpy as np
from uedge import *


def get_dr_plate(r):
    dr = np.zeros(len(r))
    for i in range(1, len(r) - 1):
        dr[i] = 0.5 * (r[i + 1] - r[i]) + 0.5 * (r[i] - r[i - 1])
    dr[0] = 2 * (r[1] - r[0])
    dr[-1] = 2 * (r[-1] - r[-2])
    return dr


def get_Q_target_proportions():
    bbb.plateflux()
    q_odata = (bbb.sdrrb + bbb.sdtrb).T
    q_idata = (bbb.sdrlb + bbb.sdtlb).T
    q1 = q_odata[0]
    q2 = q_odata[1]
    q3 = q_idata[1]
    q4 = q_idata[0]
    r1 = com.yyrb.T[0]
    r2 = com.yyrb.T[1]
    r3 = com.yylb.T[1]
    r4 = com.yylb.T[0]
    dr1 = get_dr_plate(r1)
    dr2 = get_dr_plate(r2)
    dr3 = get_dr_plate(r3)
    dr4 = get_dr_plate(r4)

    P1 = np.sum(q1[1:-1] * dr1[1:-1])
    P2 = np.sum(q2[1:-1] * dr2[1:-1])
    P3 = np.sum(q3[1:-1] * dr3[1:-1])
    P4 = np.sum(q4[1:-1] * dr4[1:-1])

    return P1, P2, P3, P4
