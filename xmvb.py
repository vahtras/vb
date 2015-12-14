#!/usr/bin/env python
"""Process and execute an XMVB input file"""
import sys, re
from util import full
#from dalton import hf
import vb

def ProcessXMVB(f):
    """Process XMVB input file"""
    #
    # Input file f read in to list of strings without : lines
    # Replase newline with separator ; for regexp tools (re.search)
    # to work
    #
    lines = ";".join(f.read().split('\n'))
    #
    # ctrl namelist
    #
    # $ctrl
    # nbasis=5 norb=5 nelectron=8 nstruc=3 nmul=1 iopt=1 iout=1 itmax=200
    # $end
    #
    pattern = r'\$ctrl(.*?)\$end'
    match = re.search(pattern, lines)
    if match:
        #print match.group(0)
        #print match.group(1)
        #
        # split on blanks or ; to obtain the assignment pairs (nbasis=5)
        #
        keywords = re.split('[ ;]', match.group(1))
        #print keywords
        ctrl = {}
        for pair in keywords:
#
# Skip empty elements
#
            if pair:
                f, w = pair.split('=')
                ctrl[f] = int(w)
    else:
        print "No match for $ctrl"
    vb.Nod.C = full.unit(ctrl["nbasis"])
    #
    # struct
    # $struct
    # 1 1 2 2 3 3 4 5
    # 1 1 2 2 3 3 4 4
    # 1 1 2 2 3 3 5 5
    # $end
    #
    pattern = r'\$struct(.*?)\$end'
    match = re.search(pattern, lines)
    structs = []
    if match:
        #print match.group(1)
        keywords = re.split('[;]', match.group(1))
        for seq in keywords:
            if seq:
                #print seq
                structs.append(ProcessStruct(seq))
    #
    # Initial structure coefficients?
    #
    return structs

def process_ctrl(lines):
    ctrl={}
    collect = False
    entries = []
    for line in lines:
        if line == '$end':
            break
        if line == '$ctrl':
            collect = True
            continue
        if collect:
            entries += line.split()    
    pairs = [word.split('=') for word  in entries if '=' in word]
    singles = {word:True  for word  in entries if '=' not in word}

    keywords = { k: w for k, w in pairs }
    keywords.update(singles)

    for k in keywords:
        try:
            iw = int(keywords[k])
            keywords[k] = iw
        except ValueError:
            pass

    return keywords 
            

def ProcessStruct(seq):
    """ Process structure from XMVB input - list of ints """
    a = []
    b = []
    iseq = []
    cseq = seq.split()
    for c in cseq:
        iseq.append(int(c)-1)
    astring = iseq[::2]
    bstring = iseq[1::2]
    dets = []
    coef = []
    dets.append(vb.Nod(astring, bstring))
    coef.append(1.0)
    #
    # If a and b are different assume singlet coupling
    #
    for a, b in zip(astring, bstring):
        if a != b:
            a2 = astring[:]
            b2 = bstring[:]
            i = astring.index(a)
            a2[i], b2[i] = b2[i], a2[i]
            dets.append(vb.Nod(a2, b2))
            coef.append(1.0)
    vbs = vb.Structure(dets, coef)
    return vbs

def main(*args, **kwargs):
    """Main VB driver"""
    print "\npyVB\n----"
    XMVB_input, Mol_input = args[:2]
    #
    # Optional suffix
    #
    try:
        XMVB_input.index('.')
    except ValueError:
        XMVB_input += ".xmvb"
    try:
        Mol_input.index('.')
    except ValueError:
        Mol_input += ".mol"
    try:
        xmvb_f = open(XMVB_input)
    except IOError:
        print "File %s not found"%XMVB_input
        sys.exit(1)
    try:
        mol_f = open(Mol_input)
        mol_f.close()
    except IOError:
        print "File %s not found"%XMVB_input
        sys.exit(1)
    #
    # Calculate Dalton HF state
    #
    hf.main(Mol_input)
    #
    # Process XMVB input
    #
    structs = ProcessXMVB(xmvb_f)
    print "\nStructures\n-------"
    for s in structs:
        print s
    xmvb_f.close()
    #
    # initial coefficients? provided list in input
    #
    if kwargs.has_key("inits"):
        if kwargs["inits"]:
            coef = kwargs["inits"]
        else:
            coef = len(structs)*[1.]
    else:
        coef = len(structs)*[1.]
    #
    # Generate VB wave function
    #
    WF = vb.WaveFunction(structs, coef, frozen=kwargs["frozen"])
    WF.Normalize()
    #print "main:WF", WF
    #
    # Optimize
    #
    #print WF.hessianmatrix().eig()
    from bfgs import pfg
    from vbscf import step, energy, gradient
    optme = pfg.pfg(step, energy, gradient)
    #
    # Initial assignment
    #
    optme.p0 = WF
    print "WF(0)", WF, "p0", optme.p0
    print "E(0)", WF.energy() + WF.Z, "f0(0)", optme.f0
    print "WFg(0)", WF.gradientvector(), "g0(0)", optme.g0
    print "Initial", optme
    #
    # Optimize
    #
    optme.bfgs(maxit=kwargs["maxit"], maxback=kwargs["maxback"])
    #
    # Final state
    #
    WF = optme.p0
    output(WF)

def output(WF):
    """Analyze"""

    for g in WF.energygrad():
        print "Grad", g
    for m in WF.vb_metric():
        print "Metric ", m
    G = WF.vb_metric_matrix()
    print "Metric", G
    print "Metric, eigenvalues", G.eig()
    print "Gradient vector", WF.gradientvector()

    WF.coef *= 2
    for m in WF.vb_metric():
        print "2*Metric ", m
    print "Gradient vector", WF.gradientvector()
    WF.Normalize()
    for m in WF.vb_metric():
        print "Metric ", m

    C = WF.C
    S = vb.Nod.S
    SC = WF.coef
    SO = WF.StructureOverlap()
    SH = WF.StructureHamiltonian()+WF.Z*SO
    W = WF.StructureWeights()
    e, V = (SH/SO).eigvec()
    print "Final orbitals", C
    print "Orbital overlap", C.T*S*C
    print "Structure coefficients", SC
    print "Structure Hamiltonian", SH, SC.T*SH*SC
    print "Structure overlap", SO
    print "Structure weights", W, W.sum()
    print "Eigenvectors/values", e, V

    print WF, WF.norm(), WF.energy(), WF.gradientvector()
    WF.Normalize()
    print "Normalized"
    print WF.norm()
    print WF, WF.norm(), WF.energy(), WF.gradientvector()



if __name__ == "__main__":
    import optparse
    Usage = "Usage: %s xmvb.input mol.input"%sys.argv[0]
    OP = optparse.OptionParser()
    OP.add_option(
        '-m', '--maxit', dest='maxit', type='int', default=10,
        help='Max number of iterations'
        )
    OP.add_option(
        '-b', '--maxback', dest='maxback', type='int', default=5,
        help='Max number of back steps'
        )
    OP.add_option(
        '-i', '--initial-structs', dest='inits', default=[],
        help='Initial stucture coefficients'
        )
    OP.add_option(
        '-f', '--frozen', dest='frozen', default=[],
        help='List of frozen orbitals'
        )
    o, a = OP.parse_args(sys.argv[1:])
    try:
        xmvb, mol = a[:]
    except IndexError:
        print Usage
    inits = []
    if o.inits:
        for i in o.inits.split():
            inits.append(float(i))
    frozen = []
    if o.frozen:
        #print o.frozen
        for i in o.frozen.split(','):
            frozen.append(int(i))
    main(
        xmvb, mol, maxit=o.maxit, maxback=o.maxback, inits=inits, frozen=frozen
        )
