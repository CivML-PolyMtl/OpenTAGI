classdef agvi
    methods (Static)
        function [mL_diag, mL_off_diag] = detach_diag_chol_vec(mL)
            n = size(mL,1);
            m = floor(sqrt(2*n));
            mL_mat = triu(ones(m));
            % form the upper triangle matrices, unpacks L column wise (ex. L_11, L_12, L_22, L_13, L_23, L_33)
            triu_ind = logical(mL_mat);
            mL_mat(triu_ind) = mL;
            mL_diag = diag(mL_mat);

            triu_ind = logical(triu(ones(m),1));
            mL_off_diag = mL_mat(triu_ind);
        end
        function mL = attach_diag_chol_vec(mL_diag, mL_off_diag)
            m = size(mL_diag,1);
            mL_mat = triu(ones(m),1);
            % form the upper triangle matrices, unpacks L column wise (ex. L_11, L_12, L_22, L_13, L_23, L_33)
            triu_ind = logical(mL_mat);
            mL_mat(triu_ind) = mL_off_diag;
            diag_ind = logical(eye(m));
            mL_mat(diag_ind) = mL_diag;
            mL = mL_mat(logical(triu(mL_mat)));
        end
        function [mLa_, SLa_, CLa] = transform_chol_vec(mLa, SLa, gpu)
            [mLa_diag, mLa_off_diag] = agvi.detach_diag_chol_vec(mLa);
            [SLa_diag, SLa_off_diag] = agvi.detach_diag_chol_vec(SLa);
            [mLa_diag_, SLa_diag_, CLa] = act.expFun(mLa_diag, SLa_diag, gpu);
            mLa_ = agvi.attach_diag_chol_vec(mLa_diag_, mLa_off_diag);
            SLa_ = agvi.attach_diag_chol_vec(SLa_diag_, SLa_off_diag);
        end
        % prior error covariance from the random vectors L
        function [mv2, Sv2] = chol_to_mv2(mL, SL)
            n = size(mL,1);
            m = floor(sqrt(2*n));
            mL_mat = triu(ones(m));
            SL_mat =  triu(ones(m));
            [ind_i, ind_j] = ind2sub(size(mL_mat), find(logical(mL_mat)));
        
            mv2 = zeros(n,1);
            Sv2 = zeros(n);
            % form the upper triangle matrices, unpacks L column wise (ex. L_11, L_12, L_22, L_13, L_23, L_33)
            mL_mat(logical(mL_mat)) = mL;
            SL_mat(logical(SL_mat)) = SL;
            
            for l = 1:size(ind_i,1)
                i = ind_i(l);
                j = ind_j(l);
                if i == j
                    cov_ij = SL_mat(:,j);
                else
                    cov_ij = zeros(m,1);
                end
                [mv2_, Sv2_] = gma.xy(mL_mat(:,i), mL_mat(:,j), SL_mat(:,i), SL_mat(:,j), cov_ij);
                mv2(l) = sum(mv2_);
                Sv2(l,l) = sum(Sv2_);
            end
        end
        function Cv2 = chol_to_Cv2(mL, SL)
            n = size(mL,1);
            Cv2 = zeros(n);

            m = floor(sqrt(2*n));
            mL_mat = triu(ones(m));
            SL_mat =  triu(ones(m));
            [triu_ind_i, triu_ind_j] = ind2sub(size(mL_mat), find(logical(mL_mat)));

            % form the upper triangle matrices, unpacks L column wise (ex. L_11, L_12, L_22, L_13, L_23, L_33)
            mL_mat(logical(mL_mat)) = mL;
            SL_mat(logical(SL_mat)) = SL;

            % for each element in l_vect
            for l_ind = 1:size(triu_ind_i,1)
                l_row_ind = triu_ind_i(l_ind);
                l_col_ind = triu_ind_j(l_ind);
                % for each WiWj_bar in the WW_bar matrix, where the indices are the columns we are extracting from the L matrix
                for ww_ind =1:size(triu_ind_i,1)
                    i = triu_ind_i(ww_ind);
                    j = triu_ind_j(ww_ind);
                    cov_lj = zeros(size(SL_mat(:,j)));
                    cov_li = zeros(size(SL_mat(:,i)));
                    % check if current l element is in either of the extracted column; if so, update cov_lj and cov_li accordingly
                    % note that current l can only be correlated to a single cell of both or either of the extracted columns
                    if l_col_ind == i
                        cov_li(l_row_ind,1) = SL_mat(l_row_ind, l_col_ind);
                    end
                    if l_col_ind == j
                        cov_lj(l_row_ind,1) = SL_mat(l_row_ind, l_col_ind);
                    end
                    Cv2(l_ind, ww_ind) = sum(gma.Cxyz(mL_mat(:,i), mL_mat(:,j), cov_li, cov_lj));
                end
            end
        end
        function [mvv, Svv] = v2_to_vv(mv2, Sv2)
            mvv = mv2;
            Svv = zeros(size(Sv2));
            % # reshape input for easier access of elements
            n = size(mv2,1);
            m = floor(sqrt(2*n));
            triu_ind = logical(triu(ones(m)));
            
            mv2_mat = zeros(m);
            Sv2_mat = zeros(m);

            mv2_mat(triu_ind) = mv2;
            Sv2_mat(triu_ind) = diag(Sv2);

            [triu_ind_i, triu_ind_j] = ind2sub(size(triu_ind), find(triu_ind));

            triu_ind = num2cell([triu_ind_i, triu_ind_j],2);

            % i, j are the indices for building the Svv matrix
            for i=1:size(triu_ind,1) % for each element in [ (0,0), (0,1), (1,1), (1,1), (1,2), (2,2) ... ]
                ind_i = triu_ind{i};
                for j=1:size(triu_ind,1) % for each element in [ (0,0), (0,1), (1,1), (1,1), (1,2), (2,2) ... ]
                    ind_j = triu_ind{j};
                    if all(ind_i == ind_j) % ex. if (0,0) == (0,0) or (0,1) == (0,1)
                        if ind_i(1) == ind_i(2) % if p == q in (p,q) then use var((Vp)^2) or equivantely var((Vq)^2)
                            Svv(i,j) = 3 * Sv2_mat(ind_i(1), ind_i(2)) + 2 * mv2_mat(ind_i(1), ind_i(2))^2;
                        else % use var((ViVj))
                            Sij = Sv2_mat(ind_i(1), ind_i(2));
                            mij = mv2_mat(ind_i(1), ind_i(2));
                            mi = mv2_mat(ind_i(1), ind_i(1));
                            mj = mv2_mat(ind_i(2), ind_i(2));
                            Svv(i,j) = Sij + mij^2/(mi * mj + mij^2) * Sij + mi * mj + mij^2;
                        end
                    else  % use cov((ViVj),(VlVm))
                        mil = mv2_mat(ind_i(1), ind_j(1));
                        mjm = mv2_mat(ind_i(2), ind_j(2));
                        mim = mv2_mat(ind_i(1), ind_j(2));
                        mjl = mv2_mat(ind_i(2), ind_j(1));
                        Svv(i,j) = mil * mjm + mim * mjl;
                    end
                end
            end
            % we calculated only the elements in the upper triangle of Svv
            % fill up Svv since it is symmetric
            Svv = Svv + triu(Svv,1)';
        end
    end
end