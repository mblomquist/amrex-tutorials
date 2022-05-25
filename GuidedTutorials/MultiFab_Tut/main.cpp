
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "Hello world from AMReX version " << amrex::Version() << "\n";

        //define a MultiFab

        int ncomp = 1;
        int ngrow = 0;
        int n_cell = 32;
        int max_grid_size = 16;


        amrex::IntVect dom_lo(0,0,0);
        amrex::IntVect dom_hi(n_cell-1,n_cell-1,n_cell-1);

        amrex::Box domain(dom_lo, dom_hi);

        amrex::BoxArray ba(domain);

        ba.maxSize(max_grid_size);

        amrex::DistributionMapping dm(ba);

        amrex::MultiFab mf(ba, dm, ncomp, ngrow); 

        //add data to the MultiFab
        amrex::RealBox real_box({0.,0.,0.},{1.,1.,1.});

        amrex::Geometry geom(domain, &real_box);

        amrex::GpuArray<amrex::Real, 3> dx = geom.CellSizeArray();

	      for(amrex::MFIter mfi(mf); mfi.isValid(); ++mfi){
	
          const amrex::Box& bx = mfi.validbox();
          const amrex::Array4<amrex::Real>& mf_array = mf.array(mfi);\

          amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){
            
            amrex::Real x = (i+0.5) * dx[0];
            amrex::Real y = (j+0.5) * dx[1];
            amrex::Real z = (k+0.5) * dx[2];

            amrex::Real rsquared = ((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5))/0.01;

            mf_array(i,j,k) = 1.0 + std::exp(-rsquared);



          });
        }

        //plot the MultiFab
        WriteSingleLevelPlotfile("plt001", mf, {"comp0"}, geom, 0., 0);

    }
    amrex::Finalize();

}

