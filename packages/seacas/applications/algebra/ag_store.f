C    Copyright(C) 1999-2020 National Technology & Engineering Solutions
C    of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
C    NTESS, the U.S. Government retains certain rights in this software.
C
C    See packages/seacas/LICENSE for details
C=======================================================================
      SUBROUTINE STORE (ISTEP, TYP, IBEGIN, IEND, NWRDS,
     &   NUMELB, IDELB, ISEVOK, VISELB, MAXNE, VARVAL, MERR)
C=======================================================================

C   --*** STORE *** (ALGEBRA) Store database variable data
C   --   Written by Amy Gilkey - revised 08/16/88
C   --
C   --STORE reads each database variable and stores it, if it is wanted,
C   --into the proper location of VARVAL.  The /VAR../ variables between
C   --IBEGIN and IEND are assumed to be ordered on IDVAR, with no
C   --repetitions.
C   --
C   --Parameters:
C   --   ISTEP  - IN  - the time step number
C   --   TYP    - IN  - the variable types
C   --   IBEGIN - IN  - the starting /VAR../ index of the variables
C   --   IEND   - IN  - the ending /VAR../ index of the variables
C   --   NWRDS  - IN  - the number of words to be read for each variable
C   --   NUMELB - IN  - the number of elements per block
C   --   ISEVOK - IN  - the element block variable truth table;
C   --                  variable i of block j exists iff ISEVOK(j,i)
C   --   MAXNE  - IN  - the VARVAL dimension (max of NUMEL and NUMNP)
C   --   VARVAL - OUT - the returned needed input variables
C   --   MERR   - OUT - error flag C   --
C   --Common Variables:
C   --   Uses IDVAR, ISTVAR of /VAR../
C   --   Uses NDBIN of /DBASE/

      PARAMETER (ICURTM = 1, ILSTTM = 2, IONETM = 3)
      include 'ag_namlen.blk'
      include 'ag_var.blk'
      include 'ag_dbase.blk'
      include 'ag_dbnums.blk'

      CHARACTER TYP
      INTEGER NUMELB(*)
      INTEGER IDELB(*)
      LOGICAL ISEVOK(NELBLK,NVAREL)
      LOGICAL VISELB(NELBLK)
      REAL VARVAL(MAXNE,*)
      INTEGER MERR
      MERR = 0

C     Read and store all global variables (if needed for the current step)
      IF (TYP .EQ. 'G') THEN
         IF (ISTVAR(ICURTM,IBEGIN) .NE. 0) THEN
            NSTO = ISTVAR(ICURTM,IBEGIN)
            ID = IDVAR(IBEGIN)
            call exggv(ndbin, istep, nvargl, varval(id,nsto), ierr)
         END IF
C     Read and store all needed nodal variables for the current step
      ELSE IF (TYP .EQ. 'N') THEN
         DO 100 NVAR = IBEGIN, IEND
            IF (ISTVAR(ICURTM,NVAR) .NE. 0) THEN
               NSTO = IABS (ISTVAR(ICURTM,NVAR))
               call exgnv(ndbin, istep, idvar(nvar), nwrds,
     &                    varval(1,nsto), ierr)
            END IF
  100    CONTINUE
C     Read and store all needed element variables for the current step
      ELSE IF (TYP .EQ. 'E') THEN
         IEL = 1
         DO 130 IELB = 1, NELBLK
            IDEB = IDELB(IELB)
            NELEM = NUMELB(IELB)
            IF (VISELB(IELB)) THEN
              DO 120 NVAR = IBEGIN, IEND
               INVAR = IDVAR(NVAR)
               IF (ISTVAR(ICURTM,NVAR) .NE. 0) THEN
                  NSTO = IABS (ISTVAR(ICURTM,NVAR))
                  IF (ISEVOK(IELB,INVAR)) THEN
                    call exgev(ndbin, istep, INVAR,
     &                IDEB, NELEM, varval(iel,nsto), ierr)
                  ELSE
C                  --Make sure values for undefined elements are zero
                     DO 110 I = IEL, IEL+NUMELB(IELB) - 1
                        VARVAL(I,NSTO) = 0.0
  110                CONTINUE
                  END IF
               END IF
  120         CONTINUE
            END IF
            IEL = IEL + NELEM
  130    CONTINUE
      ELSE
         WRITE(*,10)
     &   'Unknown Type in Subroutine Store: TYP =  ',typ
 10      format(A,A)
         MERR = 1
      END IF

      RETURN
      END
