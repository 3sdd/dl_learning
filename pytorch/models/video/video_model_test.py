import torchvision

if __name__=="__main__":
    r3d_18=torchvision.models.video.r3d_18()
    print(r3d_18)
    mc3_18=torchvision.models.video.mc3_18()
    print(mc3_18)
    r2plus1d_18=torchvision.models.video.r2plus1d_18()
    print(r2plus1d_18)