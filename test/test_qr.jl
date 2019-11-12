using LinearAlgebra

figure()
for k in 1:1000
    v = randn(2)
    Q,R = qr(v)
    plot([0,Q[1,1]],[0,Q[2,1]],"b-",alpha=0.3)
    plot([0,Q[1,2]],[0,Q[2,2]],"r-",alpha=0.3)
end

figure()
for k in 1:1000
    v = randn(3)
    Q,R = qr(v)
    plot3D([0,Q[1,1]],[0,Q[2,1]],[0,Q[3,1]],"b-",alpha=0.3)
    plot3D([0,Q[1,2]],[0,Q[2,2]],[0,Q[3,2]],"r-",alpha=0.3)
end



figure()
for k in 1:10000
    v = randn(3,3)
    Q,R = qr(v)
    plot3D([Q[1,1]],[Q[2,1]],[Q[3,1]],"bo",alpha=0.2)
    plot3D([Q[1,2]],[Q[2,2]],[Q[3,2]],"ro",alpha=0.2)
end

figure()
for k in 1:10000
    v = randn(3,3)
    Q,R = qr(v)
    Q = Q*Diagonal(sign.(diag(R)))
    plot3D([Q[1,1]],[Q[2,1]],[Q[3,1]],"bo",alpha=0.2)
    plot3D([Q[1,2]],[Q[2,2]],[Q[3,2]],"ro",alpha=0.2)
end