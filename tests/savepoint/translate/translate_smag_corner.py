from f90nml import Namelist

from ndsl import StencilFactory
from ndsl.stencils.testing import Grid
from pyfv3.stencils import SmagorinskyDiffusion
from pyfv3.testing import TranslateDycoreFortranData2Py


class TranslateSmagCorner(TranslateDycoreFortranData2Py):
    """Translate test for Smagorinsky Diffusion"""

    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ) -> None:
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "smag_c": {},
            "nord_col": {},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {"smag_c": {}}

    def compute_from_storage(self, inputs):
        nonzero_nord_k = -1
        nord_column = inputs["nord_col"][:]
        for k, value in enumerate(nord_column):
            if value > 0:
                nonzero_nord_k = k
                break

        # Only apply SmagorinskyDiffusion at "higher levels" where `nord != 0`
        self._high_k_stencil_factory = self.stencil_factory.restrict_vertical(
            k_start=nonzero_nord_k
        )

        self._smagorinsky_diffusion = SmagorinskyDiffusion(
            self._high_k_stencil_factory,
            self.grid.quantity_factory,
            self.grid.grid_data,
            self.config.grid_type,
        )

        self._smagorinsky_diffusion(
            **inputs,
            dx=self.grid.dx,
            dxc=self.grid.dxc,
            dy=self.grid.dy,
            dyc=self.grid.dyc,
            rarea=self.grid.rarea,
            rarea_c=self.grid.rarea_c,
        )
        return inputs
