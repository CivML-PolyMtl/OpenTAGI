% ml = randn(6,1);
% ml = [1;2;3;4;5;6];
ml = [1;2;3];
sl = ml/2;

% [m_d, m_nd] = agvi.detach_diag_chol_vec(ml);
% [s_d, s_nd] = agvi.detach_diag_chol_vec(sl);
% ml_ = agvi.attach_diag_chol_vec(m_d, m_nd);
% sl_ = agvi.attach_diag_chol_vec(s_d, s_nd);

% [mLa_, SLa_, CLa] = agvi.transform_chol_vec(ml, sl, false);

[mv2, sv2] = agvi.chol_to_mv2(ml, sl);
% note when comparing results to python script that the if size(ml,1)==6 rows 2,3 are flipped and cols 2,3 are flipped...
% this is because we iterate over triu_ind column-wise in matlab and row-wise in python
% Cv2 = agvi.chol_to_Cv2(ml, sl);

[mvv, svv] = agvi.v2_to_vv(mv2, sv2); % not tested when l_vect.dim == 6
