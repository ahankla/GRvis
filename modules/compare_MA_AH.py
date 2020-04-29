from metric_utils import *
from raw_data_utils import *
import sys
sys.path.append('../scripts/')
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

metric_AH = kerrschild(x1v, x2v, x3v)
metric_MA = kerr(x1v, x2v)

print("Compare metric coefficients")
print((metric_AH.g00[0, :, :] == metric_MA.g00).all())
print((metric_AH.g10[0, :, :] == metric_MA.g10).all())
print((metric_AH.g03[0, :, :] == metric_MA.g03).all())
print((metric_AH.g11[0, :, :] == metric_MA.g11).all())
print((metric_AH.g22[0, :, :] == metric_MA.g22).all())


print("The falses")
print((metric_AH.g33[0, :, :] == metric_MA.g33).all())
print((metric_AH.g13[0, :, :] == metric_MA.g13).all())
print("With allclose: ")
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
gamma_AH = metric_AH.get_normal_frame_gamma((vel1, vel2, vel3))
fourvec_MA = fourvector(x1v, x2v, vel1, vel2, vel3, Bcc1, Bcc2, Bcc3)
print(np.allclose(gamma_AH[0, :, :], fourvec_MA.gamma))
print(np.allclose(metric_AH.usq, fourvec_MA.usq))
print("AH usq max: {}".format(np.max(metric_AH.usq)))
print("MA usq max: {}".format(np.max(fourvec_MA.usq)))
print(np.allclose(metric_AH.test, fourvec_MA.test))
print(np.allclose(metric_AH.uu1, fourvec_MA.v1))
print(np.allclose(metric_AH.uu2, fourvec_MA.u2))
print(np.allclose(metric_AH.uu3, fourvec_MA.u3))

print(metric_AH.elementwise.shape)
print(metric_AH.elementwise2.shape)
print((metric_AH.elementwise2 == metric_AH.elementwise).all())
print((metric_AH.elementwise3 == metric_AH.elementwise).all())
print(np.allclose(metric_AH.elementwise2, fourvec_MA.test))
print(np.allclose(metric_AH.comp2, fourvec_MA.comp2))
print(np.allclose(metric_AH.test2, fourvec_MA.test))


print(np.allclose(metric_AH.elementwise3, metric_AH.elementwise4))
print(np.allclose(metric_AH.elementwise3, metric_AH.elementwise5))
print(np.allclose(metric_AH.usq_MA, fourvec_MA.usq))
