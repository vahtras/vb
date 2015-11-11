"""Test module of VB derivatives with numerical differentiation"""
import os
import numpy as np
import random, unittest
from daltools import one
from daltools.util import full, timing
import vb
from num_diff import findif as fd

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
      vb.Nod.C=full.matrix((2,2)).random()
      vb.Nod.S=one.read("OVERLAP",tmp('AOONEINT')).unpack().unblock()
      vb.Nod.h=one.read("ONEHAMI",tmp('AOONEINT')).unpack().unblock()
      vb.Nod.Z=one.readhead(tmp('AOONEINT'))['potnuc']
      ion=vb.Structure( [vb.Nod([0],[0]),vb.Nod([1],[1])], [1,1] )
      cov=vb.Structure( [vb.Nod([0],[1]),vb.Nod([1],[0])], [1,1] )
      cg=random.random()
      cu=random.random()
      Sab=0.13533528
      Ng2=1/(2*(1+Sab))
      Nu2=1/(2*(1-Sab))
      cion=cg*Ng2+cu*Nu2
      ccov=cg*Ng2-cu*Nu2
      self.WF=vb.WaveFunction([ion,cov],[cion,ccov],tmpdir='/tmp')
      #
      # Threshold for numerical differentiation
      #
      self.delta=1e-4
      self.t_setup.stop()

   #@unittest.skip('not working')
   def test_energy_orb_hessian(self):
      """Energy orbital Hessian"""
      _, _, numorbhess = self.WF.numenergyhess()
      _, _, anaorbhess = self.WF.energyhess()
      _, _, np.testing.assert_allclose(numorbhess, anaorbhess, self.delta)

   def test_energy_mixed_hessian(self):
      """Energy mixed Hessian"""
      _, nummixhess, _ = self.WF.numenergyhess(self.delta)
      _, anamixhess, _ = self.WF.energyhess()
      np.testing.assert_allclose(nummixhess, anamixhess, rtol=self.delta , atol=self.delta)

   def test_energy_struct_hessian(self):
      """Energy structure Hessian"""
      numstrhess, _, _ = self.WF.numenergyhess(self.delta)
      anastrhess, _, _ = self.WF.energyhess()
      np.testing.assert_allclose(numstrhess, anastrhess, rtol=self.delta, atol=self.delta)

   def test_energy_orb_gradient(self):
      """Energy orbital gradient"""
      _, numorbgr = self.WF.numenergygrad()
      _, anaorbgr = self.WF.energygrad()
      np.testing.assert_allclose(numorbgr, anaorbgr, rtol=self.delta)

   def test_energy_struct_gradient(self):
      """Energy structure gradient"""
      numstrgr, _ = self.WF.numenergygrad()
      anastrgr, _ = self.WF.energygrad()
      np.testing.assert_allclose(numstrgr, anastrgr, rtol=self.delta)

   def test_norm_orb_hessian(self):
      """Norm orbital hessian"""
      _, _, numorbhess = self.WF.numnormhess()
      _, _, anaorbhess = self.WF.normhess()
      _, _, np.testing.assert_allclose(numorbhess, anaorbhess, self.delta)

   def test_norm_mixed_hessian(self):
      """Norm mixed hessian"""
      _, nummixhess, _ = self.WF.numnormhess()
      _, anamixhess, _ = self.WF.normhess()
      np.testing.assert_allclose(nummixhess, anamixhess, self.delta)

   def test_norm_struct_hessian(self):
      """Norm structure hessian"""
      numstrhess = fd.clhess(self.WF, 'norm', 'coef')()
      anastrhess, _, _ = self.WF.normhess()
      np.testing.assert_allclose(numstrhess, anastrhess, self.delta)

   def test_norm_orb_gradient(self):
      """Norm orbital gradient"""
      _, numorbgr = self.WF.numnormgrad()
      _, anaorbgr = self.WF.normgrad()
      np.testing.assert_allclose(numorbgr, anaorbgr, rtol=self.delta)

   def test_norm_struct_gradient(self):
      """Norm structure gradient"""
      numstrgr = fd.clgrad(self.WF, 'norm', 'coef')()
      anastrgr, _ = self.WF.normgrad()
      np.testing.assert_allclose(numstrgr, anastrgr, rtol=self.delta)

   def test_nel(self):
      """Number of electrons"""
      nel=self.WF.nel()
      self.failUnlessAlmostEqual(nel,2.0,6,'Wrong electron number %g != %g'%(nel,2))

if __name__ == "__main__":
   unittest.main()
