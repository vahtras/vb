"""Test module of VB derivatives with numerical differentiation"""
import os
import numpy as np
import random, unittest
from daltools import one
from daltools.util import full, timing
import vb

class test_vb(unittest.TestCase):
   def setUp(self):
      """
       Model the fci result as a wb wave function

       WF = cg(1sg|1sg) + cu(1su|1su)

       1sg=Ng(a+b)  norm Ng=1/sqrt(2(1+Sab))
       1su=Nu(a-b)  norm Nu=1/sqrt(2(1-Sab))

       WF = cg*Ng**2(a+b|a+b) + cu*Nu**2(a-b|a-b)
	  = (cg*Ng**2+cu*Nu**2)[(a|a) + (b|b)]
	  + (cg*Ng**2-cu*Nu**2)[(a|b) + (b|a)]
      """
      self.t_setup=timing.timing("setUp")
      #
      # Dalton setup
      #
      self.dalexe="dalton.x"
      self.tmp="/tmp/"
      def tmp(fil):
	 return os.path.join(self.tmp, fil)
      self.molinp=tmp("MOLECULE.INP")
      self.dalinp=tmp("DALTON.INP")
      self.one=tmp("AOONEINT")
      self.two=tmp("AOTWOINT")
#
# A common dalton input to calculate integrals
#
      dalinp=open(self.dalinp,'w')
      dalinp.write("""**DALTON
.INTEGRAL
**END OF
""")
      dalinp.close()
      #
      # Molecule input
      #
      molinp=open(self.molinp,'w')
      molinp.write("""BASIS
STO-3G
1
2
    1    0         A
        1.    2    1    1
A   0.0  0.0  0.0
B   0.0  0.0  0.7428
""")
      molinp.close()
#
# Run dalton
#
      from subprocess import call
      cmd="cd %s; BASDIR=$(dirname $(which %s))/basis %s" % (self.tmp, self.dalexe, self.dalexe)
      print "COMMAND", cmd
      returncode = call(cmd, stdout=open('/dev/null'), shell=True)
      assert returncode == 0, "returncode=%d" % returncode
#
# Setup VB wave function
#
      vb.nod.C=full.matrix((2,2)).random()
      vb.nod.S=one.read("OVERLAP",tmp('AOONEINT')).unpack().unblock()
      vb.nod.h=one.read("ONEHAMI",tmp('AOONEINT')).unpack().unblock()
      vb.nod.Z=one.readhead(tmp('AOONEINT'))['potnuc']
      ion=vb.structure( [vb.nod([0],[0]),vb.nod([1],[1])], [1,1] )
      cov=vb.structure( [vb.nod([0],[1]),vb.nod([1],[0])], [1,1] )
      cg=random.random()
      cu=random.random()
      Sab=0.13533528
      Ng2=1/(2*(1+Sab))
      Nu2=1/(2*(1-Sab))
      cion=cg*Ng2+cu*Nu2
      ccov=cg*Ng2-cu*Nu2
      self.WF=vb.wavefunction([ion,cov],[cion,ccov],tmpdir='/tmp')
      #print self.WF
      #print self.WF.structs
      #print "Norm:     ",self.WF.norm()
      #
      # Threshold for numerical differentiation
      #
      self.delta=1e-4
      self.t_setup.stop()
      #print self.t_setup

   @unittest.skip('oo failing')
   def test_energyhess(self):
      print "Energy hessian...",
      self.t_numenergyhess=timing.timing("numenergyhess")
      (NumStrHess,NumMixHess,NumOrbHess)=self.WF.numenergyhess()
      self.t_numenergyhess.stop()
      self.t_energyhess=timing.timing("energyhess")
      (AnaStrHess,AnaMixHess,AnaOrbHess)=self.WF.energyhess()
      self.t_energyhess.stop()
      ResStrHess=(NumStrHess-AnaStrHess).norm2()
      ResMixHess=(NumMixHess-AnaMixHess).norm2()
      ResOrbHess=(NumOrbHess-AnaOrbHess).norm2()
      self.failUnless(ResStrHess<self.delta,'Energy structure hessian numeric test failed %g > %g'%(ResStrHess,self.delta))
      self.failUnless(ResMixHess<self.delta,'Energy mixed hessian numeric test failed %g > %g'%(ResMixHess,self.delta))
      self.failUnless(ResOrbHess<self.delta,'Energy orbital hessian numeric test failed %g > %g'%(ResOrbHess,self.delta))
      print "OK"
      print self.t_numenergyhess
      print self.t_energyhess

   def test_energygrad(self):
      print "Energy gradient...",
      self.t_numenergygrad=timing.timing("numenergygrad")
      (NumStrGrad,NumOrbGrad)=self.WF.numenergygrad()
      self.t_numenergygrad.stop()
      self.t_energygrad=timing.timing("energygrad")
      (AnaStrGrad,AnaOrbGrad)=self.WF.energygrad()
      self.t_energygrad.stop()
      ResStrGrad=(NumStrGrad-AnaStrGrad).norm2()
      ResOrbGrad=(NumOrbGrad-AnaOrbGrad).norm2()
      self.failUnless(ResStrGrad<self.delta,'Energy structure gradient numeric test failed %g > %g'%(ResStrGrad,self.delta))
      self.failUnless(ResOrbGrad<self.delta,'Energy orbital gradient numeric test failed %g > %g'%(ResOrbGrad,self.delta))
      print "OK"
      print self.t_numenergygrad
      print self.t_energygrad

   def test_normhess(self):
      print "Norm hessian...",
      self.t_numnormhess=timing.timing("numnormhess")
      (NumStrHess,NumMixHess,NumOrbHess)=self.WF.numnormhess()
      self.t_numnormhess.stop()
      self.t_normhess=timing.timing("normhess")
      (AnaStrHess,AnaMixHess,AnaOrbHess)=self.WF.normhess()
      self.t_normhess.stop()
      ResStrHess=(NumStrHess-AnaStrHess).norm2()
      ResMixHess=(NumMixHess-AnaMixHess).norm2()
      ResOrbHess=(NumOrbHess-AnaOrbHess).norm2()
      self.failUnless(ResStrHess<self.delta,'Norm structure hessian numeric test failed %g > %g'%(ResStrHess,self.delta))
      self.failUnless(ResMixHess<self.delta,'Norm mixed hessian numeric test failed %g > %g'%(ResMixHess,self.delta))
      self.failUnless(ResOrbHess<self.delta,'Norm orbital hessian numeric test failed %g > %g'%(ResOrbHess,self.delta))
      print "OK"
      print self.t_numnormhess
      print self.t_normhess

   def test_norm_orb_gradient(self):
      """Norm orbital gradient"""
      _, numorbgr = self.WF.numnormgrad()
      _, anaorbgr = self.WF.normgrad()
      np.testing.assert_allclose(numorbgr, anaorbgr, rtol=self.delta)

   def test_norm_struct_gradient(self):
      """Norm structure gradient"""
      numstrgr, _ = self.WF.numnormgrad()
      anastrgr, _ = self.WF.normgrad()
      np.testing.assert_allclose(numstrgr, anastrgr)

   def test_nel(self):
      """Number of electrons"""
      self.t_nel=timing.timing("nel")
      nel=self.WF.nel()
      self.t_nel.stop()
      #print "Electrons:     ",nel
      self.failUnlessAlmostEqual(nel,2.0,6,'Wrong electron number %g != %g'%(nel,2))
      print "OK"
      print self.t_nel

#  def test_print(self):
#     print "Printing"
#     print self.WF
#     print self.WF.structs
#     print "Norm:     ",self.WF.norm()

if __name__ == "__main__":
   unittest.main()
