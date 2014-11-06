import math
from dalton import one,two
from util import full
from bfgs import pfg
import vb

def step(wf,stepsize,dwf):
   dS=dwf[:len(wf.structs)]
   ao,mo=wf.C.shape
   dC=dwf[len(wf.structs):].reshape((ao,len(wf.opt)),order='Fortran')
   new=vb.wavefunction(wf.structs,wf.coef,frozen=wf.frozen)
   new.coef=wf.coef+stepsize*dS
   new.C[:,:]=wf.C[:,:]
   new.C[:,wf.opt] += stepsize*dC
   new.Normalize()
   return new
   
   
def energy(wf):
   return wf.energy() + wf.Z
def gradient(wf):
   sg,og=wf.energygrad()
   og1=og.flatten('F')
   g=full.matrix(sg.size+og1.size)
   g[:sg.size]=sg
   g[sg.size:]=og1
   return g
def hessian(wf):
   sh,mh,oh=wf.energyhess()
   mo,ao,ls=mh.shape
   lc=mo*ao
   mh1=mh.reshape((lc,ls),order='Fortran')
   oh1=oh.reshape((lc,lc),order='Fortran')
   hdim=lc+ls
   h=full.matrix((hdim,hdim))
   h[:ls,:ls]=sh
   h[ls:,:ls]=mh1
   h[:ls,ls:]=mh1.T
   h[ls:,ls:]=oh1
   return h

def main():
   #
   # Setup dalton input
   #
   tmp="/tmp"
   dalinp=open(tmp+"/DALTON.INP",'w')
   dalinp.write("""**DALTON
.INTEGRAL
**END OF
   """)
   dalinp.close()

   molinp=open(tmp+"/MOLECULE.INP",'w')
   molinp.write("""BASIS
STO-3G      
1
2
    1    0     
        1.    2    1    1
H   0.0  0.0   0.0
H   0.0  0.0   1.39839723155
    1    1
1.0       1.0
""")
   molinp.close()
   #
   # Run dalton
   #
   dalexe="/opt/dalton/bin/dalton.x"
   basdir="/opt/dalton/basis/"
   from subprocess import call
   cmd="cd %s; BASDIR=%s %s"%(tmp,basdir,dalexe)
   returncode = call(cmd,shell=True)
   #
   # setup VB object
   #
   vb.nod.C=full.matrix((2,2)); vb.nod.C[0,0]=1; vb.nod.C[1,1]=1
   vb.nod.S=one.read("OVERLAP",tmp+'/AOONEINT').unpack().unblock()
   iona=vb.structure( [vb.nod([0],[0])], [1] )
   ionb=vb.structure( [vb.nod([1],[1])], [1] )
   is2=1/math.sqrt(2)
   cov=vb.structure( [vb.nod([0],[1]),vb.nod([1],[0])], [is2,is2] )
   cg=0.95861136 ;cu=-.28471785
   #cg=1.0 ;cu=0.0
   Sab=0.13533528
   Ng2=1/(2*(1+Sab))
   Nu2=1/(2*(1-Sab))
   cion=cg*Ng2+cu*Nu2
   ccov=cg*Ng2-cu*Nu2
   WF=vb.wavefunction([cov,iona,ionb],[ccov,cion,cion],tmpdir='/tmp')


   # test update
   if 0:
      print WF,WF.C
      G=gradient(WF)
      print "G",G
      WF1=step(WF,0.1,G)
      print "W0",WF,"W1",WF1
      print WF is WF1


   if 1:
      VBSCF=pfg.pfg(step,energy,gradient)
      VBSCF.p0=WF
      VBSCF.bfgs(maxback=5,initstep=1)
      WF=VBSCF.p0
      E=VBSCF.f0
      sg,og=WF.energygrad(); print "Gradient",sg,og
      WF.Normalize()
      C=WF.C
      S=vb.nod.S
      SC=WF.coef
      SO=WF.StructureOverlap()
      SH=WF.StructureHamiltonian()+WF.Z*SO
      W=WF.StructureWeights()
      e,V=(SH/SO).eigvec()
      print "Final orbitals",C
      print "Orbital overlap",C.T*S*C
      print "Structure coefficients",SC
      print "Structure Hamiltonian",SH,SC.T*SH*SC
      print "Structure overlap",SO
      print "Structure weights",W,W.sum()
      print "Eigenvectors/values",e,V


   if 0:
      VBSCF=pfg.pfg(step,energy,gradient,hessian)
      VBSCF.p0=WF
      #VBSCF.step(0.1,-VBSCF.g0)
      #print VBSCF.p0, VBSCF.p
      #print VBSCF.p0 is VBSCF.p
      VBSCF.newton(maxit=20)
      print VBSCF.p0 is VBSCF.p
      #cg,og=WF.energygrad()
      r
      #print VBSCF.g0, cg,og
      #ch,mh,oh=WF.energyhess()
      #print VBSCF.H0, ch,mh,oh

if __name__ == "__main__":
   main()
