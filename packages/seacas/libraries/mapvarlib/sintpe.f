C Copyright(C) 1999-2020 National Technology & Engineering Solutions
C of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
C NTESS, the U.S. Government retains certain rights in this software.
C
C See packages/seacas/LICENSE for details

C=======================================================================
*DECK,SINTPE
      SUBROUTINE SINTPE (ICONA,SOLENA,ISRCHR,NISR,RSRCHR,NRSR,
     &                   SOLEB,IDBLK,XB,YB,ZB,
     &                   ICONB,ITT,IBLK,TIMES,CENTER,
     &                   ISTP,IST,INSUB,ICOMPL,DUME)

C     ******************************************************************

C     SUBROUTINE TO CONTROL INTERPOLATION OF ELEMENT TRANSFORMED
C     INTO NODAL RESULTS FROM MESH-A TO MESH-B
C     INTERPOLATED SOLUTION IS PASSED OUT TO BE RETRANSFORMED
C     INTO ELEMENT RESULTS AND THEN WRITTEN TO MESH-C EXODUS FILE

C     Calls subroutines SHAPEF

C     Called by MAPVAR

C     ******************************************************************

C ICONA   INT   Connectivity of donor mesh (1:nelnda,1:numeba)
C SOLENA  REAL  Element variables at nodes for donor mesh
C ISRCHR  INT   Donor mesh element in which point is found
C NISR    INT   Dimension for ISRCHR
C RSRCHR  REAL  Isoparametric coordinates of point in ISRCHR
C NRSR    INT   Dimension for RSRCHR
C SOLEB   REAL  Element variables for RECIPIENT Mesh
C IDBLK   INT   Recipient mesh element block I.D.
C XB,etc  REAL  Recipient mesh coordinates
C ICONB   INT   Recipient mesh connectivity
C ITT     INT   Truth table
C IM      INT   Mesh block being processed (not block I.D.)
C TIMES   REAL  Array of times associated with time step
C CENTER  REAL  Recipient mesh element centroid coordinates
C ISTP    INT   Time step
C INSUB   INT   Number of times into subroutine for this element block
C               1-first time; >1-second,etc time;
C               used to control mapping for many element blocks to one
C ICOMPL  INT   Flag for completion of mapping; used in many to one
C               0-incomplete; 1-complete

C     ******************************************************************

      include 'aexds1.blk'
      include 'aexds2.blk'
      include 'amesh.blk'
      include 'bmesh.blk'
      include 'ebbyeb.blk'
      include 'ex2tp.blk'
      include 'tapes.blk'

      DIMENSION TIMES(*), CENTER(NUMEBB,*)
      DIMENSION ICONA(NELNDA,*), SOLENA(NODESA,NVAREL)
      DIMENSION ITT(NVAREL,*)
      DIMENSION SOLEB(NUMEBB,*), XB(*), YB(*), ZB(*)
      DIMENSION ISRCHR(NISR,*), RSRCHR(NRSR,*), ICONB(nelndb,*)
      DIMENSION SOLN(27), XX(27), YY(27), ZZ(27), DUME(*)

C     ******************************************************************

      DO 10 IVAR = 1,NVAREL
        IF (ITT(IVAR,IBLK) .EQ. 0)GO TO 10

C Initialize SOLEB if first time in subroutine for this element block
C After first time into subroutine
C retrieve SOLEB from storage in EXODUS

        IF (INSUB .EQ. 1)THEN
          CALL INIELT(SOLEB,IVAR,TIMES,ISTP,IDBLK,CENTER,DUME)
        ELSE
          CALL EXGEV(NTP4EX,IST,IVAR,IDBLK,NUMEBB,SOLEB(1,IVAR),IERR)
        END IF

C Loop on centroids in recipient mesh

        DO 30 I = 1,NUMEBB
          IF (ISRCHR(1,I) .NE. 0)THEN

C Set parameters for element in donor mesh

            S = RSRCHR(5,I)
            T = RSRCHR(6,I)
            R = 0.
            NNODES = 4
            DO 20 J = 1,NNODES
              INODE = ICONA(J,ISRCHR(1,I))
              SOLN(J) = SOLENA(INODE,IVAR)
 20         CONTINUE

C Shape function to evaluate interpolation

            CALL SHAPEF (3,S,T,R,SOLN,BVALUE)
            SOLEB(I,IVAR) = BVALUE
          ELSE
            if (ivar .eq. 1 .and. icompl .eq. 1) then
              write (nout, 9000) i, idblk
              write (ntpout, 9000) i, idblk
 9000         FORMAT('SINTPE: No donor element found for ',
     *          'recipient Element', i9, ' in block ', i9)
            end if
          END IF
 30     CONTINUE

C  If there is more searching to do (i.e. many blocks to one)
C  use EXODUS as temporary storage
C  don't bother to make adjustments to element variables yet

        IF (ICOMPL .NE. 1)THEN
          CALL EXPEV(NTP4EX,IST,IVAR,IDBLK,NUMEBB,SOLEB(1,IVAR),IERR)
        ELSE

C Do needed modification to element variable values
C write element vars out to EXODUS data base

C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C ELMASS is special

          IF (NAMVAR(nvargp+IVAR)(1:6) .EQ. 'ELMASS')THEN

C ELMASS was changed to nodal density prior to processing.
C need to go back from density to element mass now

            DO 100 IEL = 1, NUMEBB
              DO 105 I = 1, NNODES
                XX(I) = XB(ICONB(I,IEL))
                YY(I) = YB(ICONB(I,IEL))
                ZZ(I) = ZB(ICONB(I,IEL))
 105          CONTINUE
              CALL VOL(ITYPE,XX,YY,ZZ,VOLUME)
              SOLEB(IEL,IVAR) = SOLEB(IEL,IVAR) * VOLUME
 100        CONTINUE
          END IF

C****************************************************************
C apply constraints to variables here as applicable

C Plastic strain (EQPS) and tearing
C must greater than or equal to 0.

          IF (NAMVAR(nvargp+IVAR)(1:4) .EQ. 'EQPS')THEN
            DO 110 IEL = 1, NUMEBB
              IF (SOLEB(IEL,IVAR) .LT. 0.)THEN
                SOLEB(IEL,IVAR) = 0.
              END IF
  110       CONTINUE
          ELSE IF (NAMVAR(nvargp+IVAR)(1:7) .EQ. 'TEARING')THEN
            DO 120 IEL = 1, NUMEBB
              IF (SOLEB(IEL,IVAR) .LT. 0.)THEN
                SOLEB(IEL,IVAR) = 0.
              END IF
  120       CONTINUE

C Hourglass forces and bulk viscosity have no meaning other than on
C the mesh from which they originated, just set them to zero.

          ELSE IF (NAMVAR(nvargp+IVAR)(1:2) .EQ. 'HG' .OR.
     &            NAMVAR(nvargp+IVAR)(1:5) .EQ. 'BULKQ')THEN
            DO 130 IEL = 1, NUMEBB
              SOLEB(IEL,IVAR) = 0.
  130       CONTINUE

C Shell element basis vectors are computed from the mesh
C It makes no sense to transfer them - set them to zero.

          ELSE IF (NAMVAR(nvargp+IVAR)(1:5) .EQ. 'BASEL')THEN
            DO 140 IEL = 1, NUMEBB
              SOLEB(IEL,IVAR) = 0.
 140        CONTINUE

C         ELSE IF (NAMVAR(nvargp+IVAR)(1:?) .EQ. ?????)THEN
C           DO ??? IEL = 1, NUMEBB
C             IF (SOLEB(IEL,IVAR) .??. ?.)THEN
C               SOLEB(IEL,IVAR) = ?.
C             END IF
C  ???      CONTINUE

          END IF
c************************************************************************

C write element variables

          CALL EXPEV(NTP4EX,IST,IVAR,IDBLK,NUMEBB,SOLEB(1,IVAR),IERR)
        END IF
   10 CONTINUE
      RETURN
      END
