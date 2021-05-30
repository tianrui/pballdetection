import utilities
import pdb


def main():
    """ small test with basic images of two classes of pool balls, with one pose for each
    """
    n = 10
    #objsdir = "~/Documents/pballdetection/assets"
    #bgpath = "~/Documents/pballdetection/bg.png"
    #datadir = "~/Documents/pballdetection/dataset"
    objsdir = "./assets"
    bgpath = "./assets/bg.png"
    datadir = "./dataset"
    
    pdb.set_trace()
    annospath = utilities.compile_img(n, objsdir, bgpath, datadir)

if __name__=="__main__":
    main()
