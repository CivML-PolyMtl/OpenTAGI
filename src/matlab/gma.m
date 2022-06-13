classdef gma
    methods (Static)
        function [mxy, Sxy] = xy(mx, my, Sx, Sy, Cxy)
            mxy = mx.*my + Cxy;
            Sxy = Sx.*Sy + Cxy.^2 + 2 * Cxy .* mx .* my + mx.^2 .* Sy + my.^2 .* Sx;
        end
        function C_x_yz = Cxyz(my, mz, Cxy, Cxz)
            C_x_yz = Cxy .* mz + Cxz .* my;
        end
    end
end