from metric_utils import *
from raw_data_utils import *
import sys
sys.path.append('../scripts/')
sys.path.append('m_athay/')
from kerrmetric import fourvector, kerr


raw_data_path = "/mnt/c/Users/liaha/scratch/"
config = "1.1.1-torus2_b-gz2"
specs = "a0beta500torBeta_br32x32x64rl2x2"
time = 26

filename = raw_data_path + config + "_" + specs + ".prim.{:05}.athdf".format(time)
raw_data = read_athdf(filename, quantities=["rho", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3"])
x1v = raw_data["x1v"]
x2v = raw_data["x2v"]
x3v = raw_data["x3v"]
vel1 = raw_data["vel1"]
vel2 = raw_data["vel2"]
vel3 = raw_data["vel3"]
Bcc1 = raw_data["Bcc1"]
Bcc2 = raw_data["Bcc2"]
Bcc3 = raw_data["Bcc3"]

r10ind = np.argmin(np.abs(x1v - 10.0))
x1vr = x1v[r10ind].reshape(1,)

metric_AH = kerrschild(x1vr, x2v, x3v)
metric_MA = kerr(x1vr, x2v)
nx3 = x3v.size
nx2 = x2v.size
nx1 = x1v.size
print(x1v.size)
print(x2v.size)
print("Compare metric coefficients")
print((metric_AH.g00[0, :, :] == metric_MA.g00).all())
print((metric_AH.g10[0, :, :] == metric_MA.g10).all())
print((metric_AH.g03[0, :, :] == metric_MA.g03).all())
print((metric_AH.g11[0, :, :] == metric_MA.g11).all())
print((metric_AH.g22[0, :, :] == metric_MA.g22).all())
print(np.allclose(metric_AH.g13[0, :, :], metric_MA.g13))
print(np.allclose(metric_AH.g33[0, :, :], metric_MA.g33))

print("Jacobian: ")
print((metric_AH.jacobian[0, :, :] == metric_MA.g).all())

print("Compare inverse metric coefficients (with allclose)")
print(np.allclose(metric_AH.G00[0, :, :], metric_MA.G00))
print(np.allclose(metric_AH.G10[0, :, :], metric_MA.G10))
print(np.allclose(metric_AH.G11[0, :, :], metric_MA.G11))
print(np.allclose(metric_AH.G22[0, :, :], metric_MA.G22))
print(np.allclose(metric_AH.G31[0, :, :], metric_MA.G31))
print(np.allclose(metric_AH.G33[0, :, :], metric_MA.G33))

print("Compare gamma (with allclose)")
print(vel1.shape)
vel1r = vel1[:, :, r10ind].reshape(nx3, nx2, 1)
vel2r = vel2[:, :, r10ind].reshape(nx3, nx2, 1)
vel3r = vel3[:, :, r10ind].reshape(nx3, nx2, 1)
gamma_AH = metric_AH.get_normal_frame_gamma((vel1r, vel2r, vel3r))
# fourvec_MA = fourvector(x1v, x2v, vel1, vel2, vel3, Bcc1, Bcc2, Bcc3)
fourvec_MA = fourvector(x1vr, x2v, vel1r, vel2r, vel3r, Bcc1, Bcc2, Bcc3)
# print("Is first term the same?")
# print(np.allclose(metric_AH.firstterm, fourvec_MA.firstterm))
# print("Is second term the same?")
# print(np.allclose(metric_AH.secondterm, fourvec_MA.secondterm))
# print("Is third term the same?")
# print(np.allclose(metric_AH.thirdterm, fourvec_MA.thirdterm))
# print("Is fourth term the same?")
# print(np.allclose(metric_AH.fourthterm, fourvec_MA.fourthterm))
print("Is usquared the same?")
print(np.allclose(metric_AH.usq, fourvec_MA.usq))
print(np.allclose(gamma_AH, fourvec_MA.gamma))
# print("AH usq max: {}".format(np.max(metric_AH.usq)))
# print("MA usq max: {}".format(np.max(fourvec_MA.usq)))
