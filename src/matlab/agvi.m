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
            mL = mL_mat(logical(triu(ones(m))));
        end

        function [mLa_, SLa_, CLa] = transform_chol_vec(mLa, SLa, gpu)
            [mLa_diag, mLa_off_diag] = agvi.detach_diag_chol_vec(mLa);
            [SLa_diag, SLa_off_diag] = agvi.detach_diag_chol_vec(SLa);
            [mLa_diag_, SLa_diag_, CLa] = act.expFun(mLa_diag, SLa_diag, gpu);
            mLa_ = agvi.attach_diag_chol_vec(mLa_diag_, mLa_off_diag);
            SLa_ = agvi.attach_diag_chol_vec(SLa_diag_, SLa_off_diag);
        end

        function [mv2_mat, Sv2_mat] = chol_vec_to_v2mat(mLa, SLa)
            [mLa_diag, mLa_off_diag] = agvi.detach_diag_chol_vec(mLa);
            [SLa_diag, SLa_off_diag] = agvi.detach_diag_chol_vec(SLa);
            [mLa_diag_, SLa_diag_, CLa] = act.expFun(mLa_diag, SLa_diag, gpu);

            m = size(mLa_diag,1);
            mv2_mat = triu(ones(m),1);
            Sv2_mat = triu(ones(m),1);
            % form the upper triangle matrices, unpacks L column wise (ex. L_11, L_12, L_22, L_13, L_23, L_33)
            triu_ind = logical(mv2_mat);
            mv2_mat(triu_ind) = mLa_off_diag;
            Sv2_mat(triu_ind) = SLa_off_diag;
            diag_ind = logical(eye(m));
            mv2_mat(diag_ind) = mLa_diag_;
            Sv2_mat(diag_ind) = SLa_diag_;
        end

        % transform from positive domain to real domain
        % 'a' here implies the variable is in the transformed space 
        function [mLa_post_, SLa_post_] = full_noiseBackwardUpdate(mLa, SLa, mLa_prior, SLa_prior, CLa_prior, mLa_post, SLa_post, gpu)
            [mLa_diag, ~] = agvi.detach_diag_chol_vec(mLa);
            [SLa_diag, ~] = agvi.detach_diag_chol_vec(SLa);

            [mLa_diag_prior, ~] = agvi.detach_diag_chol_vec(mLa_prior);
            [SLa_diag_prior, ~] = agvi.detach_diag_chol_vec(SLa_prior);

            [mLa_diag_post, mLa_off_diag] = agvi.detach_diag_chol_vec(mLa_post);
            [SLa_diag_post, SLa_off_diag] = agvi.detach_diag_chol_vec(SLa_post);

            [deltaMLz, deltaSLz] = tagi.noiseBackwardUpdate(mLa_diag_prior, SLa_diag_prior, CLa_prior,...
                                                                mLa_diag_post, SLa_diag_post, gpu);
            
            mLa_diag = mLa_diag + deltaMLz;
            SLa_diag = SLa_diag + deltaSLz;

            mLa_post_ = agvi.attach_diag_chol_vec(mLa_diag, mLa_off_diag);
            SLa_post_ = agvi.attach_diag_chol_vec(SLa_diag, SLa_off_diag);
        end

        % prior error covariance from the random vectors L
        function [mv2, Sv2] = chol_to_v2(mL, SL)
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

        function Sv = v2_to_Sv(mv2)
            n = size(mv2,1);
            m = floor(sqrt(2*n));
            triu_ind = logical(triu(ones(m)));

            Sv = zeros(m);
            Sv(triu_ind) = mv2;
            Sv = Sv + triu(Sv,1)';
        end

        function [m_post, S_post] = gauss_cond(m_prior, S_prior, my, Sy, Cy, y)
            Sy_inv = inv(Sy);
            m_post = m_prior + Cy * Sy_inv * (y - my);
            S_post = S_prior - Cy * Sy_inv * Cy';
        end

        function [mvv, Svv] = v_to_vv(mv, Sv)
            m = size(mv,1);
            n = m*(m + 1)/2;
            mvv = zeros(n,1);
            Svv = zeros(n);

            triu_ind = logical(triu(ones(m)));
            [triu_ind_i, triu_ind_j] = ind2sub(size(triu_ind), find(triu_ind));
            triu_ind = num2cell([triu_ind_i, triu_ind_j],2);

            for i=1:size(triu_ind,1) % for each element in [ (0,0), (0,1), (1,1), (1,1), (1,2), (2,2) ... ]
                ind_i = triu_ind{i};
                for j=1:size(triu_ind,1) % for each element in [ (0,0), (0,1), (1,1), (1,1), (1,2), (2,2) ... ]
                    ind_j = triu_ind{j};
                    if all(ind_i == ind_j) % ex. if (0,0) == (0,0) or (0,1) == (0,1)
                        mi = mv(ind_i(1));
                        mj = mv(ind_i(2));
                        Si = Sv(ind_i(1), ind_i(1));
                        Sj = Sv(ind_i(2), ind_i(2));
                        Cij = Sv(ind_i(1), ind_i(2));
                        [mvv(i), Svv(i,j)] = gma.xy(mi, mj, Si, Sj, Cij);
                    else
                        mi = mv(ind_i(1));
                        mj = mv(ind_i(2));
                        ml = mv(ind_j(1));
                        mm = mv(ind_j(2));
                        Sil = Sv(ind_i(1), ind_j(1));
                        Sim = Sv(ind_i(1), ind_j(2));
                        Sjl = Sv(ind_i(2), ind_j(1));
                        Sjm = Sv(ind_i(2), ind_j(2));
                        Svv(i,j) = gma.Cabcd(mi, mj, ml, mm, Sil, Sim, Sjl, Sjm);
                    end
                end
            end
        end

        function [m_, S_] = RTS_update(m_pr, S_pr, mf, Sf, mb, Sb, Cf)
            if isnan(rcond(Sb))
                a = 10000;
            end
            K = Cf * inv(Sb);
            m_ = m_pr + K * (mf - mb);
            S_ = S_pr + K * (Sf - Sb) * K';
        end
            
        function [deltaMz, deltaSz, ml_post, Sl_post] = full_noiseUpdate4regression(mz, Sz, mLa_, SLa_, y)
            % retrieve prior parameters for v2
            [mv2, Sv2] = agvi.chol_to_v2(mLa_, SLa_);
            Cv2 = agvi.chol_to_Cv2(mLa_, SLa_);
            % retrieve prior parameters for vv
            [mvv, Svv] = agvi.v2_to_vv(mv2, Sv2);
            % retrieve prior covariance for v
            Sv = agvi.v2_to_Sv(mv2);
            % reshape the 1D array containing diaognal elements of a matrix into a diagonal matrix
            Sz = diag(Sz);
            % construct the hidden mean vector, cov matrix, and cross-cov matrix
            mh = [mz;zeros(size(mz,1),1)];
            Sh = blkdiag(Sz, Sv);
            Chy = [Sz;Sv];

            % use the observation model y = z + v to get the mean and variance of the observation  
            my = mz;
            Sy = Sz + Sv;

            % get the posterior mean and covariance of the hidden var
            [mh, Sh] = agvi.gauss_cond(mh, Sh, my, Sy, Chy, y);

            mz_post = mh(1:size(mz,1));
            mv_post = mh(size(mz,1)+1:end);
            Sz_post = Sh(1:size(Sz,1),1:size(Sz,2));
            Sv_post = Sh(size(Sz,1)+1:end,size(Sz,2)+1:end);

            [mvv_post, Svv_post] = agvi.v_to_vv(mv_post, Sv_post);

            [mv2_post, Sv2_post] = agvi.RTS_update(mv2, Sv2, mvv_post, Svv_post, mvv, Svv, Sv2);
            [ml_post, Sl_post] = agvi.RTS_update(mLa_, diag(SLa_), mv2_post, Sv2_post, mv2, Sv2, Cv2);

            deltaMz = mz_post - mz;
            deltaSz = diag(Sz_post) - diag(Sz);
            Sl_post = diag(Sl_post);
            % deltaML = ml_post - mLa_;
            % deltaSL = Sl_post - SLa_;
        end
    end
end