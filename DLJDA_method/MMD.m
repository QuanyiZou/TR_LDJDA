function M = MMD(src_X, tar_X, src_labels, Y_tar_pseudo)
% Construct MMD matrix
	X = [src_X,tar_X];
	[m,n] = size(X);
	ns = size(src_X,2);
	nt = size(tar_X,2);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	C = length(unique(src_labels));

	%%% M0
	M = e * e' * C;  %multiply C for better normalization

	%%% Mc
	N = 0;
	if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
		for c = reshape(unique(src_labels),1,C)
			e = zeros(n,1);
			e(src_labels==c) = 1 / length(find(src_labels==c));
			e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
			e(isinf(e)) = 0;
			N = N + e*e';
		end
	end

	M = M + N;
	M = M / norm(M,'fro');
end